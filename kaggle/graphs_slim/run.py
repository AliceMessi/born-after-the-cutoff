"""Private Kaggle kernel (CPU only): code graphs WITHOUT node text (licence-safe release
format) for the 519 evaluation instances. Our own harvest only; no Competition Data."""
import gzip
import json
import os
import subprocess
import sys
sys.path.insert(0, "/kaggle/working")
from common import OUT, P, record_env, save_runs, setup, sh

setup()
record_env({"job": "graphs_slim"})
sys.path[:0] = [f"{P}/loc", f"{P}/graph", f"{P}/harvest"]
import snapshots as S
from build_graph import build
from data import load
from paths import NON_CORE_RE

ids = set(open("data/v0.1/eval_ids.txt").read().split())
insts = sorted((r for r in load() if r["instance_id"] in ids), key=lambda r: (r["repo"], r["merged_at"]))
dest = f"{OUT}/graphs"
os.makedirs(dest, exist_ok=True)
cache, done = {}, 0
for r in insts:
    try:
        root = S.checkout(r["repo"], r["base_commit"])
        ls = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout
        blobs = {ln.partition("\t")[2]: ln.split()[2] for ln in ls.splitlines() if ln.endswith(".py")}
        files = sorted(f for f in blobs if not NON_CORE_RE.search(f) and (root / f).exists())
        g = build(root, files, r["repo"], r["base_commit"], cache=cache, blob_of=blobs)
    except Exception as e:
        print("skip", r["instance_id"], type(e).__name__, str(e)[:100], flush=True)
        continue
    for n in g["nodes"]:
        n.pop("text", None)  # upstream source code is not redistributed
    g["graph"]["note"] = "node text omitted; rebuild with graph/build_graph.py at the pinned commit"
    with gzip.open(f"{dest}/{r['instance_id']}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(g, fh)
    done += 1
    if done % 50 == 0:
        print(done, "graphs", flush=True)
print("graphs written:", done, flush=True)
sh(f"du -sh {dest}", check=False)
