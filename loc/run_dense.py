"""Run the dense baseline B2 on instances (needs an embeddings server).

  python -I loc/run_dense.py --model embeddinggemma-2 --served-id <id> --ids-file data/v0.1/eval_ids.txt

Function units = top-level functions, classes (header part) and methods, as in
B1. Files are ranked by their best unit (MaxP); files without units fall back
to an embedding of their first UNIT_CHARS characters.
Writes data/v0.1/runs/B2-<model>.jsonl (resumable).
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
from dense import Encoder, dot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["embeddinggemma-2", "jina-code"])
    ap.add_argument("--served-id", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--ids-file", required=True)
    args = ap.parse_args()
    enc = Encoder(args.model, args.served_id, args.base_url)
    out_path = V01 / "runs" / f"B2-{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["instance_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
    insts = sorted((r for r in load() if r["instance_id"] in keep and r["instance_id"] not in done),
                   key=lambda r: (r["repo"], r["merged_at"]))
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(insts, 1):
            snap = B.Snapshot(S.checkout(r["repo"], r["base_commit"]))
            keys, docs, owner = [], [], []
            for f in snap.files:
                units = B._units(snap.text(f))
                if not units:
                    keys.append(f"{f}::<module>")
                    docs.append(enc.doc(f, snap.text(f)))
                    owner.append(f)
                for q, src in units:
                    keys.append(f"{f}::{q}")
                    docs.append(enc.doc(f"{f} {q}", src))
                    owner.append(f)
            vecs = enc.embed(docs)
            qv = enc.query(r["problem_statement"])
            scores = [dot(qv, v) for v in vecs]
            order = sorted(range(len(keys)), key=lambda i: -scores[i])
            funcs = [keys[i] for i in order if not keys[i].endswith("::<module>")][:50]
            files, seen = [], set()
            for i in order:
                if owner[i] not in seen:
                    seen.add(owner[i])
                    files.append(owner[i])
                if len(files) == 50:
                    break
            fh.write(json.dumps({"instance_id": r["instance_id"], "n_files": len(snap.files),
                                 "files": files, "functions": funcs}) + chr(10))
            fh.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(insts)} instances, {time.time() - t0:.0f}s, api calls {enc.calls}", flush=True)
    print(f"done {len(insts)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
