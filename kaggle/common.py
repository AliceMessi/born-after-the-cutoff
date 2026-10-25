"""Shared bootstrap for our private Kaggle kernels (copied next to each kernel's run.py
by kaggle/push.py). Finds the private dataset, unpacks the project, sets env vars so
git stores and caches live in /tmp (not saved as output), and records the environment."""

import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

W = "/kaggle/working"
P = "/tmp/project"  # not under /kaggle/working, so it is not saved as output
OUT = f"{W}/out"


def setup() -> None:
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(f"{P}/loc/llm.py"):
        zips = glob.glob("/kaggle/input/**/project.zip", recursive=True)
        dirs = [d for d in glob.glob("/kaggle/input/**/project", recursive=True) if os.path.exists(f"{d}/loc/llm.py")]
        if dirs:
            shutil.copytree(dirs[0], P)
        elif zips:
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall("/tmp")
        else:
            raise SystemExit("project not found under /kaggle/input")
    os.environ["AGENTLOC_REPO_STORE"] = "/tmp/agentloc_repos"
    os.environ["AGENTLOC_EMB_CACHE"] = "/tmp/agentloc_emb.sqlite"
    os.chdir(P)


def sh(cmd: str, check: bool = True) -> int:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def record_env(extra: dict | None = None) -> None:
    info = {"python": sys.version}
    for cmd, key in (("git --version", "git"), ("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "gpu")):
        try:
            info[key] = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception as e:  # pragma: no cover
            info[key] = str(e)
    for mod in ("torch", "transformers", "sentence_transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    info.update(extra or {})
    with open(f"{OUT}/env.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info), flush=True)


def save_runs(names: list[str]) -> None:
    for n in names:
        src = f"{P}/data/v0.1/runs/{n}"
        if os.path.exists(src):
            shutil.copy(src, f"{OUT}/{n}")
            print("saved", n, sum(1 for _ in open(src, encoding="utf-8")), "rows", flush=True)
