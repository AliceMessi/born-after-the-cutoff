"""LLM localisation methods P (probe) and A (Agentless-style) over an
OpenAI-compatible endpoint (LM Studio locally, Ollama on Kaggle).

  python -I loc/run_llm.py --method A --model gemma-4-e2b-it-qat --tag e2b --ids-file data/v0.1/eval_ids.txt

Do not run on evaluation ids before PREREG.md is committed (pilot ids only).
Writes data/v0.1/runs/<method>-<tag>.jsonl (resumable) with ranked files,
ranked functions (A), token counts and latency. Predictions are normalised
against the snapshot (loc/normalize.py); unparseable outputs give empty lists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines as B  # noqa: E402
import snapshots as S  # noqa: E402
from data import V01, load, read_jsonl  # noqa: E402
from llm import chat  # noqa: E402
from normalize import files_from_text, funcs_from_text  # noqa: E402
from repo import skeleton  # noqa: E402

SYSTEM = "You are a precise software engineer. Answer only in the requested format."
PROBE = ("Repository: {repo} (Python).\nGitHub issue:\n{issue}\n\n"
         "Without seeing the code, list the 5 file paths (relative to the repository root) most likely to need "
         "changes to fix this issue, most likely first, one per line.")
A1 = ("Repository: {repo}\nGitHub issue:\n{issue}\n\nPython source files:\n{listing}\n\n"
      "Which files must be edited to fix the issue? Return the 5 most likely paths from the list above, most "
      "likely first, one per line, nothing else.")
A2 = ("GitHub issue:\n{issue}\n\nSkeletons of candidate files:\n{skel}\n\n"
      "Which functions or methods must be edited? Return up to 10 entries, most likely first, one per line, "
      "using the file paths from the ### headers above, formatted exactly as "
      "{example}::ClassName.method or {example}::function_name.")
ISSUE_CHARS = 6000
LISTING_CHARS = 40000  # ~10k tokens; longer listings are truncated in B1 order (budget ablation knob)


def run_one(r: dict, method: str, model: str, base_url: str, reasoning: str | None) -> dict:
    snap = B.Snapshot(S.checkout(r["repo"], r["base_commit"]))
    issue = r["problem_statement"][:ISSUE_CHARS]
    msgs = lambda u: [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]  # noqa: E731
    if method == "P":
        c = chat(msgs(PROBE.format(repo=r["repo"], issue=issue)), model=model, base_url=base_url,
                 max_tokens=200, reasoning_effort=reasoning)
        return {"files": files_from_text(c["text"], snap.files), "calls": [c], "raw": [c["text"][:600]]}
    listing = "\n".join(snap.files)
    if len(listing) > LISTING_CHARS:  # keep the most BM25-relevant files when the list is too long
        order = B.rank_bm25_files(snap, issue)
        keep, size = [], 0
        for f in order:
            size += len(f) + 1
            if size > LISTING_CHARS:
                break
            keep.append(f)
        listing = "\n".join(sorted(keep))
    c1 = chat(msgs(A1.format(repo=r["repo"], issue=issue, listing=listing)), model=model, base_url=base_url,
              max_tokens=200, reasoning_effort=reasoning)
    files = files_from_text(c1["text"], snap.files)
    skel = "\n\n".join(f"### {f}\n{skeleton(snap.root, f)}" for f in files[:3])
    example = files[0] if files else "path/to/file.py"  # pilot: a placeholder path was copied verbatim
    c2 = chat(msgs(A2.format(issue=issue, skel=skel, example=example)), model=model, base_url=base_url,
              max_tokens=300, reasoning_effort=reasoning)
    units = {f: [q for q, _ in B._units(snap.text(f))] for f in files[:3]}
    funcs = funcs_from_text(c2["text"], units, snap.files, bare_ok=True)
    return {"files": files, "functions": funcs, "calls": [c1, c2], "raw": [c1["text"][:600], c2["text"][:600]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["P", "A"], required=True)
    ap.add_argument("--model", required=True, help="served model id")
    ap.add_argument("--tag", required=True, help="short model tag for the output file, e.g. e2b")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--backend", choices=["openai", "ollama"], default=None,
                    help="default: $AGENTLOC_LLM_BACKEND or openai")
    ap.add_argument("--thinking", action="store_true", help="leave thinking on (default: reasoning_effort=none)")
    args = ap.parse_args()
    if args.backend:
        import os
        os.environ["AGENTLOC_LLM_BACKEND"] = args.backend
    out = V01 / "runs" / f"{args.method}-{args.tag}{'-think' if args.thinking else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {x["instance_id"] for x in read_jsonl(out)} if out.exists() else set()
    keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
    insts = sorted((r for r in load() if r["instance_id"] in keep and r["instance_id"] not in done),
                   key=lambda r: (r["repo"], r["merged_at"]))
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(insts, 1):
            try:
                res = run_one(r, args.method, args.model, args.base_url, None if args.thinking else "none")
            except Exception as e:  # API/snapshot failure: record an empty prediction, never skip silently
                res = {"files": [], "functions": [], "calls": [], "raw": [], "error": f"{type(e).__name__}: {e}"[:300]}
            calls = res.pop("calls")
            res.update(instance_id=r["instance_id"], model=args.model,
                       prompt_tokens=sum(c.get("prompt_tokens") or 0 for c in calls),
                       completion_tokens=sum(c.get("completion_tokens") or 0 for c in calls),
                       reasoning_tokens=sum(c.get("reasoning_tokens") or 0 for c in calls),
                       seconds=round(sum(c.get("seconds") or 0 for c in calls), 2))
            fh.write(json.dumps(res) + chr(10))
            fh.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(insts)}, {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(insts)} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
