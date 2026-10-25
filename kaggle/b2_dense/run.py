"""Private Kaggle kernel: dense baseline B2 (EmbeddingGemma 2, 256-d MRL) on the
evaluation + pilot ids. Our own harvest only; no Competition Data."""
import sys
sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, "/kaggle/src")
from common import OUT, P, record_env, save_runs, setup, sh

setup()
sh("pip install -q sentence-transformers==6.2.0 transformers==5.20.2")
record_env({"model": "google/embeddinggemma-2", "dtype": "float32", "truncate_dim": 256})
ids = open("data/v0.1/eval_ids.txt").read().split() + open("data/v0.1/pilot_ids.txt").read().split()
open("data/v0.1/b2_ids.txt", "w").write("\n".join(dict.fromkeys(ids)) + "\n")
sh("python loc/run_dense.py --model embeddinggemma-2 --served-id google/embeddinggemma-2 "
   "--backend st --ids-file data/v0.1/b2_ids.txt 2>&1 | tail -40", check=False)
save_runs(["B2-embeddinggemma-2.jsonl"])
sh(f"ls -la {OUT}")
# keep the content-hash embedding cache for later kernels (G2 node search reuses it)
sh(f"cp /tmp/agentloc_emb.sqlite {OUT}/emb_embeddinggemma2.sqlite && ls -la {OUT}", check=False)
