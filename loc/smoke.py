"""End-to-end smoke test: one harvested instance -> BM25 + Gemma 4 -> metrics.

Methods (all training-free):
  bm25        files ranked by BM25(issue, path + content); functions by BM25 over AST units
  probe       issue text ONLY, no repository shown: the model guesses file paths
              (memorization probe in the spirit of "The SWE-Bench Illusion")
  agentless   hierarchical prompt: core-file list -> top-5 files -> skeletons -> top-10 functions

Usage (LM Studio server on :1234 with the model loaded):
  python -I loc/smoke.py --ids openai__openai-agents-python-2860 --model <lmstudio-model-id>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repo as R  # noqa: E402
from bm25 import BM25  # noqa: E402
from llm import chat  # noqa: E402
from metrics import report  # noqa: E402

DATA = HERE.parent / "data" / "dryrun"
PATH_RE = re.compile(r"[\w./-]+\.py\b")
FUNC_RE = re.compile(r"([\w./-]+\.py)::([\w.]+)")


def load(ids: list[str]) -> list[tuple[dict, dict]]:
    out = []
    for i in ids:
        stem = i.rsplit("-", 1)[0]
        cands = {json.loads(x)["instance_id"]: json.loads(x)
                 for x in (DATA / f"{stem}.candidates.jsonl").read_text(encoding="utf-8").split(chr(10)) if x.strip()}
        golds = {json.loads(x)["instance_id"]: json.loads(x)
                 for x in (DATA / f"{stem}.gold.jsonl").read_text(encoding="utf-8").split(chr(10)) if x.strip()}
        out.append((cands[i], golds[i]))
    return out


def resolve_paths(text: str, files: list[str]) -> list[str]:
    """Map model-written paths onto real files (exact, then unique suffix)."""
    out = []
    for p in PATH_RE.findall(text):
        p = p.lstrip("./")
        hit = p if p in files else next((f for f in files if f.endswith("/" + p)), None)
        if hit and hit not in out:
            out.append(hit)
    return out


def issue_text(c: dict, limit: int = 6000) -> str:
    return f"{c['issue_title']}\n\n{c['issue_body']}"[:limit]


def run(c: dict, g: dict, model: str, base_url: str) -> dict:
    root = R.snapshot(g["repo"], g["base_commit"])
    files = R.core_files(root)
    gold_f, gold_fn = set(g["gold_files"]), set(g["gold_functions"])
    issue = issue_text(c)
    res = {"instance_id": g["instance_id"], "n_core_files": len(files),
           "gold_files": sorted(gold_f), "gold_functions": sorted(gold_fn), "flags": g["flags"]}

    # --- BM25 baseline
    bm = BM25([f + "\n" + R.read(root, f) for f in files])
    bm_files = [files[i] for i in bm.rank(issue)[:10]]
    units = [(f"{f}::{q}", src) for f in files for q, src in R.functions(root, f)]
    bmu = BM25([u + "\n" + s for u, s in units])
    bm_funcs = [units[i][0] for i in bmu.rank(issue)[:10]]
    res["bm25"] = {"files": bm_files[:5], "file": report(bm_files, gold_f), "func": report(bm_funcs, gold_fn)}

    sys_msg = {"role": "system", "content": "You are a precise software engineer. Answer only in the requested format."}

    # --- memorization probe: no repository content at all
    p = chat([sys_msg, {"role": "user", "content":
              f"Repository: {g['repo']} (Python).\nGitHub issue:\n{issue}\n\n"
              "Without seeing the code, list the 5 file paths (relative to the repository root) "
              "most likely to need changes to fix this issue, most likely first, one per line."}],
             model=model, base_url=base_url, max_tokens=200)
    probe_files = resolve_paths(p["text"], files)
    res["probe"] = {"files": probe_files, "file": report(probe_files, gold_f),
                    "tokens": [p["prompt_tokens"], p["completion_tokens"]], "seconds": p["seconds"]}

    # --- Agentless-style hierarchical localization
    listing = "\n".join(files)
    a1 = chat([sys_msg, {"role": "user", "content":
               f"Repository: {g['repo']}\nGitHub issue:\n{issue}\n\nPython source files:\n{listing}\n\n"
               "Which files must be edited to fix the issue? Return the 5 most likely paths from the list "
               "above, most likely first, one per line, nothing else."}],
              model=model, base_url=base_url, max_tokens=200)
    a_files = resolve_paths(a1["text"], files)
    skel = "\n\n".join(f"### {f}\n{R.skeleton(root, f)}" for f in a_files[:3])
    a2 = chat([sys_msg, {"role": "user", "content":
               f"GitHub issue:\n{issue}\n\nSkeletons of candidate files:\n{skel}\n\n"
               "Which functions or methods must be edited? Return up to 10 entries, most likely first, one "
               "per line, formatted exactly as path/to/file.py::ClassName.method or path/to/file.py::function."}],
              model=model, base_url=base_url, max_tokens=300)
    a_funcs = []
    for path, q in FUNC_RE.findall(a2["text"]):
        rp = resolve_paths(path, files)
        key = f"{rp[0]}::{q}" if rp else f"{path}::{q}"
        if key not in a_funcs:
            a_funcs.append(key)
    res["agentless"] = {
        "files": a_files, "funcs": a_funcs[:10],
        "file": report(a_files, gold_f), "func": report(a_funcs, gold_fn),
        "tokens": [a1["prompt_tokens"], a1["completion_tokens"], a2["prompt_tokens"], a2["completion_tokens"]],
        "seconds": round(a1["seconds"] + a2["seconds"], 2),
        "raw_step1": a1["text"][:400],
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--out", default=str(DATA / "smoke_results.jsonl"))
    args = ap.parse_args()
    with open(args.out, "a", encoding="utf-8") as fh:
        for c, g in load(args.ids):
            r = run(c, g, args.model, args.base_url)
            r["model"] = args.model
            fh.write(json.dumps(r) + "\n")
            print(json.dumps({k: r[k] for k in ("instance_id", "n_core_files", "gold_files")}))
            for m in ("bm25", "probe", "agentless"):
                f = r[m]["file"]
                print(f"  {m:9} file hit@1={f['hit@1']:.0f} hit@5={f['hit@5']:.0f} mrr={f['mrr']:.2f}"
                      + (f"  func hit@5={r[m]['func']['hit@5']:.0f} acc_all@10={r[m]['func']['acc_all@10']:.0f}" if "func" in r[m] else "")
                      + (f"  tokens={r[m].get('tokens')} s={r[m].get('seconds')}" if m != "bm25" else ""))


if __name__ == "__main__":
    main()
