"""Run baseline RM (repo-map personalised PageRank) on instances. CPU only, no LLM.

  python -I loc/run_rm.py --ids-file data/v0.1/eval_ids.txt
Writes data/v0.1/runs/RM-pagerank.jsonl (resumable): ranked core files and functions.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "harvest"))
import snapshots as S  # noqa: E402
from baseline_rm import rank_repo_map  # noqa: E402
from build_graph import build  # noqa: E402
from data import V01, load, read_jsonl  # noqa: E402
from paths import NON_CORE_RE, is_core_py  # noqa: E402

GRAPH_CACHE: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    args = ap.parse_args()
    out = V01 / "runs" / "RM-pagerank.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {x["instance_id"] for x in read_jsonl(out)} if out.exists() else set()
    keep = set(Path(args.ids_file).read_text(encoding="utf-8").split())
    insts = sorted((r for r in load() if r["instance_id"] in keep and r["instance_id"] not in done),
                   key=lambda r: (r["repo"], r["merged_at"]))
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(insts, 1):
            try:
                root = S.checkout(r["repo"], r["base_commit"])
                p = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "HEAD"], cwd=root,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
                blobs = {ln.partition("\t")[2]: ln.split()[2] for ln in p.stdout.splitlines() if ln.endswith(".py")}
                files = sorted(f for f in blobs if not NON_CORE_RE.search(f) and (root / f).exists())
                g = build(root, files, r["repo"], r["base_commit"], cache=GRAPH_CACHE, blob_of=blobs)
            except Exception as e:  # fetch/parse failure: log and continue
                print(f"  skip {r['instance_id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            ranked_files, order, n_seeds = rank_repo_map(g, r["problem_statement"])
            mod_of = {x["file"]: x["id"] for x in g["nodes"] if x["type"] == "module"}
            node = {x["id"]: x for x in g["nodes"]}
            funcs = []
            for i in order:
                x = node[i]
                if x["type"] != "module" and is_core_py(x["file"]):
                    funcs.append(f"{x['file']}::{i[len(mod_of[x['file']]) + 1:]}")
                if len(funcs) == 50:
                    break
            core_files = [f for f in ranked_files if is_core_py(f)][:50]
            fh.write(json.dumps({"instance_id": r["instance_id"], "files": core_files, "functions": funcs,
                                 "n_seeds": n_seeds}) + chr(10))
            fh.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(insts)}, {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(insts)} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
