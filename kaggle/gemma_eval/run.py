"""Private Kaggle kernel (T4): Gemma 4 localization runs on our own harvest.
CONFIG (injected by kaggle/push.py --config) sets the model, methods and id files.
No Competition Data. Runs resume from previous outputs mounted as kernel sources."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
sys.path.insert(0, "/kaggle/working")
from common import OUT, P, record_env, save_runs, setup, sh

URL = "http://127.0.0.1:11434"
setup()
os.makedirs("data/v0.1/runs", exist_ok=True)
for name in ["B2-embeddinggemma-2.jsonl"] + [f"{m}-{CONFIG['tag']}.jsonl" for m in ("P", "A", "G1", "G2")]:
    for f in glob.glob(f"/kaggle/input/**/{name}", recursive=True)[:1]:   # seeds and partial runs to resume
        shutil.copy(f, f"data/v0.1/runs/{name}")
        print("restored", name, "from", f, flush=True)
for f in glob.glob("/kaggle/input/**/emb_embeddinggemma2.sqlite", recursive=True)[:1]:
    shutil.copy(f, "/tmp/agentloc_emb.sqlite")
if "G2" in CONFIG["methods"]:
    sh("pip install -q sentence-transformers==6.2.0 transformers==5.20.2")
sh("apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq zstd > /dev/null 2>&1; which zstd")
sh("curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.11 sh > /tmp/ollama_install.log 2>&1; tail -2 /tmp/ollama_install.log")
sh("which ollama && ollama --version")
env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_CONTEXT_LENGTH=str(CONFIG.get("num_ctx", 16384)),
           OLLAMA_KEEP_ALIVE="-1", CUDA_VISIBLE_DEVICES=CONFIG.get("llm_gpus", "0"))
srv = subprocess.Popen(["ollama", "serve"], env=env, stdout=open("/tmp/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen(f"{URL}/api/tags", timeout=2)
        break
    except Exception:
        time.sleep(2)
model = CONFIG["model"]
sh(f"ollama pull {model} > /tmp/pull.log 2>&1; tail -2 /tmp/pull.log")
info = json.loads(urllib.request.urlopen(f"{URL}/api/tags").read())
digest = [m.get("digest") for m in info.get("models", []) if model in (m.get("name"), m.get("model"))]
record_env({"job": f"gemma-{CONFIG['tag']}", "config": CONFIG, "digest": digest,
            "ollama": subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()})
common = f"--backend ollama --base-url {URL} --model {model} --tag {CONFIG['tag']}"
deadline = time.time() + CONFIG.get("budget_hours", 10.5) * 3600
for m in CONFIG["methods"]:
    if time.time() > deadline:
        print("budget reached before", m, flush=True)
        break
    ids = CONFIG["g2_ids"] if m == "G2" else CONFIG["ids"]
    if m in ("P", "A"):
        cmd = f"python loc/run_llm.py --method {m} {common} --ids-file {ids}"
    elif m == "A-masked":  # PREREG H5: identifier-masked issues
        cmd = f"python loc/run_llm.py --method A --mask {common} --ids-file {ids}"
    elif m == "G1":
        cmd = f"python loc/run_g1.py {common} --ids-file {ids}"
    else:
        cmd = f"python loc/run_g2.py {common} --ids-file {ids} --embed google/embeddinggemma-2 --embed-device cuda:1"
    left = max(60, int(deadline - time.time()))
    sh(f"timeout {left}s {cmd} 2>&1 | tail -3", check=False)   # resumable: partial rows are kept
    save_runs([f"A-{CONFIG['tag']}-masked.jsonl" if m == "A-masked" else f"{m}-{CONFIG['tag']}.jsonl"])
shutil.copy("/tmp/ollama.log", f"{OUT}/ollama.log")
srv.terminate()
