"""Graph diagnostics on benchmark instances (no LLM):

* graph size and call-resolution rate per snapshot (core + test files);
* gold coverage: share of gold functions that exist as graph nodes;
* reachability: share of gold functions within 1 / 2 undirected hops of the
  B1 (BM25) top-10 function seeds, and the size of that candidate pool.
  This bounds what graph expansion (method G1) can add over BM25 alone.

  python -I graph/reachability.py --ids-file data/v0.1/eval_ids.txt --limit 150 --seed 0
Writes data/v0.1/results/graph_reachability.jsonl (resumable) and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "loc"))
sys.path.insert(0, str(ROOT / "harvest"))
import baselines as B  # noqa: E402
import snapshots as S  # noqa: E402
from build_graph import build, gold_to_ids  # noqa: E402
from data import V01, load, read_jsonl  # noqa: E402
from paths import NON_CORE_RE  # noqa: E402


def hops(adj: dict[str, set[str]], seeds: list[str], k: int) -> set[str]:
    seen = set(seeds)
    q = deque((s, 0) for s in seeds)
    while q:
        n, d = q.popleft()
        if d == k:
            continue
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                q.append((m, d + 1))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    keep = [x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        keep = random.Random(args.seed).sample(keep, min(args.limit, len(keep)))
    keep = set(keep)
    out = V01 / "results" / "graph_reachability.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["instance_id"] for r in read_jsonl(out)} if out.exists() else set()
    b1 = {r["instance_id"]: r for r in read_jsonl(V01 / "runs" / "B1-bm25.jsonl")}
    insts = sorted((r for r in load() if r["instance_id"] in keep and r["instance_id"] not in done
                    and r["instance_id"] in b1 and r["gold_functions"]),
                   key=lambda r: (r["repo"], r["merged_at"]))
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(insts, 1):
            root = S.checkout(r["repo"], r["base_commit"])
            files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py"))
            files = [f for f in files if not NON_CORE_RE.search(f)]
            tb = time.time()
            g = build(root, files, r["repo"], r["base_commit"])
            build_s = time.time() - tb
            adj = defaultdict(set)
            for e in g["edges"]:
                if e["type"] in ("calls", "inherits", "contains"):
                    adj[e["source"]].add(e["target"])
                    adj[e["target"]].add(e["source"])
            gold_ids = [x for x in gold_to_ids(r["gold_functions"], g) if x]
            seeds = [x for x in gold_to_ids(b1[r["instance_id"]]["functions"][:10], g) if x]
            row = {"instance_id": r["instance_id"], "repo": r["repo"], "build_s": round(build_s, 1),
                   "nodes": g["graph"]["stats"]["nodes"], "edges": g["graph"]["stats"]["edges"],
                   "calls_resolved": g["graph"]["stats"].get("calls_resolved", 0),
                   "calls_unresolved": g["graph"]["stats"].get("calls_unresolved", 0),
                   "gold_n": len(r["gold_functions"]), "gold_in_graph": len(gold_ids), "seeds": len(seeds)}
            gs = set(gold_ids)
            symbols = {x["id"] for x in g["nodes"] if x["type"] != "module"}
            for k in (0, 1, 2):
                pool = hops(adj, seeds, k) & symbols  # candidate functions/classes/methods
                row[f"pool_{k}"] = len(pool)
                row[f"gold_reach_{k}"] = len(gs & pool)
            fh.write(json.dumps(row) + chr(10))
            fh.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(insts)}, {time.time() - t0:.0f}s", flush=True)
    rows = [x for x in read_jsonl(out) if x["instance_id"] in keep]
    if not rows:
        return
    tot_gold = sum(x["gold_n"] for x in rows)
    summ = {
        "instances": len(rows),
        "median_nodes": sorted(x["nodes"] for x in rows)[len(rows) // 2],
        "median_build_s": sorted(x["build_s"] for x in rows)[len(rows) // 2],
        "call_resolution_rate": round(sum(x["calls_resolved"] for x in rows)
                                      / max(1, sum(x["calls_resolved"] + x["calls_unresolved"] for x in rows)), 3),
        "gold_in_graph": round(sum(x["gold_in_graph"] for x in rows) / tot_gold, 3),
    }
    for k in (0, 1, 2):
        summ[f"gold_reachable_{k}hop"] = round(sum(x[f"gold_reach_{k}"] for x in rows) / tot_gold, 3)
        summ[f"any_gold_reachable_{k}hop"] = round(sum(x[f"gold_reach_{k}"] > 0 for x in rows) / len(rows), 3)
        summ[f"median_pool_{k}hop"] = sorted(x[f"pool_{k}"] for x in rows)[len(rows) // 2]
    (V01 / "results" / "graph_reachability_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
