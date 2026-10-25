"""Method G1: graph-expanded candidate pool + one listwise Gemma rerank.

Seeds: top-k functions from B1 (BM25) and, when available, B2 (dense) runs.
Pool:  seeds plus their 1-hop neighbours over calls/inherits/contains edges in
       the snapshot's code graph (graph/build_graph.py), capped at --pool units,
       BM25 order breaking ties.
Rerank: the model sees the issue and, for each candidate, its id, file and
       signature line (plus the first lines of the body), and returns up to 10
       ids. Files are ranked by the first appearance of their functions, then
       by B1 file order.

  python -I loc/run_g1.py --model gemma-4-e2b-it-qat --tag e2b --ids-file data/v0.1/eval_ids.txt
Do not run on evaluation ids before PREREG.md is committed (pilot ids only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "harvest"))
import snapshots as S  # noqa: E402
from build_graph import build  # noqa: E402
from data import V01, load, read_jsonl  # noqa: E402
from llm import chat  # noqa: E402
from normalize import files_from_text, funcs_from_text  # noqa: E402
from paths import NON_CORE_RE, is_core_py  # noqa: E402

PROMPT = ("GitHub issue:\n{issue}\n\nCandidate functions (id | first lines):\n{cands}\n\n"
          "Which of these functions must be edited to fix the issue? Return up to 10 ids from the list, most likely "
          "first, one per line, exactly as written before the first '|'.")
GRAPH_CACHE: dict = {}


def blob_map(root: Path) -> dict[str, str]:
    p = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "HEAD"], cwd=root,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    out = {}
    for line in p.stdout.splitlines():
        meta, _, path = line.partition("\t")
        if path.endswith(".py"):
            out[path] = meta.split()[2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--backend", choices=["openai", "ollama"], default=None,
                    help="default: $AGENTLOC_LLM_BACKEND or openai")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--dense-run", default="B2-embeddinggemma-2")
    args = ap.parse_args()
    if args.backend:
        import os
        os.environ["AGENTLOC_LLM_BACKEND"] = args.backend
    out = V01 / "runs" / f"G1-{args.tag}.jsonl"
    done = {x["instance_id"] for x in read_jsonl(out)} if out.exists() else set()
    keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
    b1 = {x["instance_id"]: x for x in read_jsonl(V01 / "runs" / "B1-bm25.jsonl")}
    dpath = V01 / "runs" / f"{args.dense_run}.jsonl"
    b2 = {x["instance_id"]: x for x in read_jsonl(dpath)} if dpath.exists() else {}
    insts = sorted((r for r in load() if r["instance_id"] in keep and r["instance_id"] not in done
                    and r["instance_id"] in b1), key=lambda r: (r["repo"], r["merged_at"]))
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(insts, 1):
            iid = r["instance_id"]
            root = S.checkout(r["repo"], r["base_commit"])
            blobs = blob_map(root)
            files = sorted(f for f in blobs if not NON_CORE_RE.search(f) and (root / f).exists())
            g = build(root, files, r["repo"], r["base_commit"], cache=GRAPH_CACHE, blob_of=blobs)
            mod_of = {x["file"]: x["id"] for x in g["nodes"] if x["type"] == "module"}
            key_of = {}  # graph id -> path::qual
            for x in g["nodes"]:
                if x["type"] != "module" and is_core_py(x["file"]):
                    key_of[x["id"]] = f"{x['file']}::{x['id'][len(mod_of[x['file']]) + 1:]}"
            id_of = {v: k for k, v in key_of.items()}
            adj = defaultdict(list)
            for e in g["edges"]:
                if e["type"] in ("calls", "inherits", "contains"):
                    adj[e["source"]].append(e["target"])
                    adj[e["target"]].append(e["source"])
            bm_order = b1[iid]["functions"]
            seeds = bm_order[: args.seeds] + (b2[iid]["functions"][: args.seeds] if iid in b2 else [])
            pool = list(dict.fromkeys(seeds))
            for s in list(pool):
                for nb in adj.get(id_of.get(s, ""), []):
                    k = key_of.get(nb)
                    if k and k not in pool:
                        pool.append(k)
            rank = {k: i for i, k in enumerate(bm_order)}
            pool = pool[: len(dict.fromkeys(seeds))] + sorted(pool[len(dict.fromkeys(seeds)):],
                                                              key=lambda k: rank.get(k, 10**6))
            pool = pool[: args.pool]
            nodes = {x["id"]: x for x in g["nodes"]}
            lines = []
            for k in pool:
                x = nodes.get(id_of.get(k, ""))
                head = " ".join((x["text"] if x else "").splitlines()[:3])[:220]
                lines.append(f"{k} | {head}")
            try:
                c = chat([{"role": "system", "content": "You are a precise software engineer. Answer only in the requested format."},
                          {"role": "user", "content": PROMPT.format(issue=r["problem_statement"][:6000],
                                                                    cands="\n".join(lines))}],
                         model=args.model, base_url=args.base_url, max_tokens=300)
                units = defaultdict(list)
                for k in pool:
                    f, _, q = k.partition("::")
                    units[f].append(q)
                funcs = [k for k in funcs_from_text(c["text"], dict(units), files, bare_ok=True) if k in pool][:10]
                err = None
            except Exception as e:  # record an empty prediction
                c, funcs, err = {"text": ""}, [], f"{type(e).__name__}: {e}"[:300]
            picks = list(funcs)
            named_files = files_from_text(c["text"], files)
            if picks or named_files:
                # a listwise reranker reorders the whole pool: model picks first, the rest in seed order (pilot)
                funcs = picks + [k for k in pool if k not in picks] if picks else []
                file_order = list(dict.fromkeys(k.partition("::")[0] for k in picks))
                file_order += [f for f in named_files if f not in file_order]
                file_order += [f for f in dict.fromkeys(k.partition("::")[0] for k in funcs) if f not in file_order]
                file_order += [f for f in b1[iid]["files"] if f not in file_order]
            else:  # PREREG §5: an unparseable or failed output is an empty ranking, never a fallback
                funcs, file_order = [], []
            fh.write(json.dumps({"instance_id": iid, "model": args.model, "files": file_order[:50],
                                 "functions": funcs[:50], "model_picks": len(picks) if not err else 0,
                                 "pool_size": len(pool),
                                 "prompt_tokens": c.get("prompt_tokens"), "completion_tokens": c.get("completion_tokens"),
                                 "seconds": c.get("seconds"), "raw": c["text"][:600], "error": err}) + chr(10))
            fh.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(insts)}, {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(insts)} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
