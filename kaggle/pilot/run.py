"""Private Kaggle kernel (T4): step-3 pilot (PREREG §6) on 30 T1 ids outside the
evaluation set. Methods P, A, G1 with Gemma 4 E2B and E4B (QAT) served by Ollama.
Our own harvest only; no Competition Data."""
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

MODELS = {"e2b": "gemma4:e2b-it-qat", "e4b": "gemma4:e4b-it-qat"}
IDS = "data/v0.1/pilot_ids.txt"
URL = "http://127.0.0.1:11434"

setup()
os.makedirs("data/v0.1/runs", exist_ok=True)
for f in glob.glob("/kaggle/input/**/B2-embeddinggemma-2.jsonl", recursive=True)[:1]:
    shutil.copy(f, "data/v0.1/runs/B2-embeddinggemma-2.jsonl")
    print("B2 seeds from", f, flush=True)
for f in glob.glob("/kaggle/input/**/emb_embeddinggemma2.sqlite", recursive=True)[:1]:
    shutil.copy(f, "/tmp/agentloc_emb.sqlite")
    print("embedding cache from", f, flush=True)
sh("pip install -q sentence-transformers==6.2.0 transformers==5.20.2")
open("data/v0.1/pilot_g2_ids.txt", "w").write(chr(10).join(open(IDS).read().split()[:15]) + chr(10))
sh(f"python loc/run_baselines.py --ids-file {IDS} 2>&1 | tail -3")

sh("apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq zstd > /dev/null 2>&1; which zstd")
sh("curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.11 sh > /tmp/ollama_install.log 2>&1; tail -3 /tmp/ollama_install.log")
sh("which ollama && ollama --version")  # fail fast if the install did not work
env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_CONTEXT_LENGTH="16384", OLLAMA_KEEP_ALIVE="-1",
           CUDA_VISIBLE_DEVICES="0")  # LLM on GPU 0; the G2 query/node encoder uses GPU 1
srv = subprocess.Popen(["ollama", "serve"], env=env, stdout=open("/tmp/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen(f"{URL}/api/tags", timeout=2)
        break
    except Exception:
        time.sleep(2)
digests = {}
for tag, model in MODELS.items():
    sh(f"ollama pull {model} > /tmp/pull_{tag}.log 2>&1; tail -2 /tmp/pull_{tag}.log")
    info = json.loads(urllib.request.urlopen(f"{URL}/api/tags").read())
    digests[model] = [m.get("digest") for m in info.get("models", []) if m.get("name") == model or m.get("model") == model]
    common = f"--backend ollama --base-url {URL} --model {model} --tag pilot-{tag} --ids-file {IDS}"
    sh(f"python loc/run_llm.py --method P {common} 2>&1 | tail -2", check=False)
    sh(f"python loc/run_llm.py --method A {common} 2>&1 | tail -2", check=False)
    sh(f"python loc/run_g1.py {common} 2>&1 | tail -2", check=False)
    g2 = common.replace(IDS, "data/v0.1/pilot_g2_ids.txt")
    sh(f"python loc/run_g2.py {g2} --embed google/embeddinggemma-2 --embed-device cuda:1 2>&1 | tail -3", check=False)
ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
record_env({"job": "pilot", "ollama": ver, "digests": digests, "num_ctx": 16384, "think": False})
save_runs([os.path.basename(f) for f in glob.glob("data/v0.1/runs/*pilot*.jsonl")] + ["B1-bm25.jsonl"])
shutil.copy("/tmp/ollama.log", f"{OUT}/ollama.log")
srv.terminate()
