"""Push queued private GPU kernels as Kaggle's 2-session limit allows.

  python -I kaggle/scheduler.py            (runs until the queue is empty; prints one line per push)

Every kernel is built with kaggle/push.py (private, pinned installs, our dataset only)
and compile-checked before it is pushed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUEUE = [
    {"slug": "gemma-12b", "tag": "12b", "model": "gemma4:12b-it-qat", "methods": ["P", "A", "G1"],
     "ids": "data/v0.1/eval_ids.txt", "g2_ids": "data/v0.1/g2_ids.txt", "budget_hours": 10.5},
    {"slug": "gemma-g2-e2b", "tag": "e2b", "model": "gemma4:e2b-it-qat", "methods": ["G2"],
     "ids": "data/v0.1/eval_ids.txt", "g2_ids": "data/v0.1/g2_ids.txt", "budget_hours": 10.5},
    {"slug": "gemma-g2-e4b", "tag": "e4b", "model": "gemma4:e4b-it-qat", "methods": ["G2"],
     "ids": "data/v0.1/eval_ids.txt", "g2_ids": "data/v0.1/g2_ids.txt", "budget_hours": 10.5},
    {"slug": "gemma-g2-12b", "tag": "12b", "model": "gemma4:12b-it-qat", "methods": ["G2"],
     "ids": "data/v0.1/eval_ids.txt", "g2_ids": "data/v0.1/g2_ids.txt", "budget_hours": 10.5},
]
WATCH = ["gemma-e4b", "gemma-e2b"]  # already running


def status(slug: str) -> str:
    p = subprocess.run(["kaggle", "kernels", "status", f"<kaggle-user>/agentloc-{slug}"], capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


def running(slugs: list[str]) -> int:
    return sum(1 for s in slugs if any(x in status(s) for x in ("RUNNING", "QUEUED")))


def push(cfg: dict) -> bool:
    args = [sys.executable, "-I", str(HERE / "push.py"), "gemma_eval", "--gpu",
            "--kernel-source", "agentloc-b2-dense", "--config", json.dumps(cfg)]
    subprocess.run(args, check=True, capture_output=True)
    src = ROOT / "data" / "kaggle_kernels" / f"agentloc-{cfg['slug']}" / "run.py"
    compile(src.read_text(encoding="utf-8"), str(src), "exec")
    p = subprocess.run(args + ["--push"], capture_output=True, text=True)
    out = p.stdout + p.stderr
    ok = "successfully pushed" in out
    print(f"{time.strftime('%H:%M')} push {cfg['slug']}: {'ok' if ok else out.strip()[-160:]}", flush=True)
    return ok


def main() -> None:
    active = list(WATCH)
    queue = list(QUEUE)
    while queue:
        if running(active) < 2:
            cfg = queue[0]
            if push(cfg):
                active.append(cfg["slug"])
                queue.pop(0)
                time.sleep(30)
                continue
        time.sleep(120)
    print(f"{time.strftime('%H:%M')} queue empty", flush=True)


if __name__ == "__main__":
    main()
