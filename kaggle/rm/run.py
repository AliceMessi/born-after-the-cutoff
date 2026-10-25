"""Private Kaggle kernel (CPU only): baseline RM (repo-map personalised PageRank, no LLM)
on the evaluation + pilot ids. Our own harvest only; no Competition Data."""
import sys
sys.path.insert(0, "/kaggle/working")
from common import OUT, P, record_env, save_runs, setup, sh

setup()
record_env({"job": "rm_pagerank"})
ids = open("data/v0.1/eval_ids.txt").read().split() + open("data/v0.1/pilot_ids.txt").read().split()
open("data/v0.1/rm_ids.txt", "w").write(chr(10).join(dict.fromkeys(ids)) + chr(10))
sh("python loc/run_rm.py --ids-file data/v0.1/rm_ids.txt 2>&1 | tail -5", check=False)
save_runs(["RM-pagerank.jsonl"])
