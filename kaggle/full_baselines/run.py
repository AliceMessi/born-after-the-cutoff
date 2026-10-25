"""Private Kaggle kernel (CPU only): retrieval baselines R0-random, R0-path and B1-bm25
on every v0.1 instance. Our own harvest only; no Competition Data."""
import sys
sys.path.insert(0, "/kaggle/working")
from common import OUT, P, record_env, save_runs, setup, sh

setup()
record_env({"job": "full_baselines"})
sh("python loc/run_baselines.py 2>&1 | tail -30", check=False)
save_runs(["R0-random.jsonl", "R0-path.jsonl", "B1-bm25.jsonl"])
sh("du -sh /tmp/agentloc_repos || true", check=False)
