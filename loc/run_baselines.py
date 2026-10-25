"""Run the retrieval baselines (R0-random, R0-path, B1-bm25) on benchmark instances.

  python -I loc/run_baselines.py [--ids-file ids.txt] [--limit N]

Writes data/v0.1/runs/<method>.jsonl with ranked files (top 50) and, for
B1, ranked functions (top 50). Resumable: instances already present in a
run file are skipped. Instances are processed repo by repo in commit order so
the snapshot store and the tokenisation cache are reused.
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

METHODS = ("R0-random", "R0-path", "B1-bm25")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", help="restrict to these instance ids (one per line)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--runs", default=str(V01 / "runs"))
    args = ap.parse_args()

    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    insts = load()
    if args.ids_file:
        keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
        insts = [r for r in insts if r["instance_id"] in keep]
    done = {}
    for m in METHODS:
        f = runs / f"{m}.jsonl"
        done[m] = {r["instance_id"] for r in read_jsonl(f)} if f.exists() else set()
    insts = [r for r in insts if any(r["instance_id"] not in done[m] for m in METHODS)]
    insts.sort(key=lambda r: (r["repo"], r["merged_at"]))
    if args.limit:
        insts = insts[: args.limit]
    handles = {m: (runs / f"{m}.jsonl").open("a", encoding="utf-8") for m in METHODS}
    t0, n = time.time(), 0
    try:
        for r in insts:
            try:
                root = S.checkout(r["repo"], r["base_commit"])
                snap = B.Snapshot(root)
            except Exception as e:  # fetch/checkout failure: log and continue
                print(f"  skip {r['instance_id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            q = r["problem_statement"]
            iid = r["instance_id"]
            out = {
                "R0-random": {"files": B.rank_random(snap, iid)[:50]},
                "R0-path": {"files": B.rank_path_bm25(snap, q)[:50]},
                "B1-bm25": {"files": B.rank_bm25_files(snap, q)[:50], "functions": B.rank_bm25_functions(snap, q)},
            }
            for m, res in out.items():
                if iid not in done[m]:
                    handles[m].write(json.dumps({"instance_id": iid, "n_files": len(snap.files), **res}) + chr(10))
                    handles[m].flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(insts)} instances, {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in handles.values():
            h.close()
    print(f"done: {n} instances in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
