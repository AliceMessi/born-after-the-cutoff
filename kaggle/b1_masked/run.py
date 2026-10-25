"""Private Kaggle kernel (CPU only): B1 BM25 on identifier-masked issues (PREREG H5),
`main` stratum (first 400 evaluation ids). Our own harvest only; no Competition Data."""
import sys
sys.path.insert(0, "/kaggle/working")
from common import OUT, P, record_env, save_runs, setup, sh

setup()
record_env({"job": "b1_masked"})
ids = open("data/v0.1/eval_ids.txt").read().split()[:400]
open("data/v0.1/main_ids.txt", "w").write(chr(10).join(ids) + chr(10))
sh("python loc/run_baselines.py --mask --ids-file data/v0.1/main_ids.txt 2>&1 | tail -3", check=False)
save_runs(["B1-bm25-masked.jsonl"])
