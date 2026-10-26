"""One lightweight poller: every 5 minutes, push the next
queued private kernel when fewer than 2 GPU sessions are busy. Logs to
data/kaggle_out/poller.log. Stops on any push error and on any kernel status other than
RUNNING / COMPLETE / QUEUED. Exits when the queue is empty and nothing is running.

  python -I kaggle/poller.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from owner import owner  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG = ROOT / "data" / "kaggle_out" / "poller.log"
BASE = {"ids": "data/v0.1/eval_ids.txt", "g2_ids": "data/v0.1/g2_ids.txt", "budget_hours": 10.5}
QUEUE = [
    dict(BASE, slug="gemma-g2-e4b", tag="e4b", model="gemma4:e4b-it-qat", methods=["G2"]),
    dict(BASE, slug="gemma-g2-12b", tag="12b", model="gemma4:12b-it-qat", methods=["G2"]),
    dict(BASE, slug="gemma-amask-e4b", tag="e4b", model="gemma4:e4b-it-qat", methods=["A-masked"],
         ids="data/v0.1/main_ids.txt", budget_hours=4),
]
TRACK = ["gemma-12b", "gemma-g2-e2b"]
OK = ("RUNNING", "COMPLETE", "QUEUED")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M')} {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + chr(10))
    print(line, flush=True)


def status(slug: str) -> str:
    p = subprocess.run(["kaggle", "kernels", "status", f"{owner()}/agentloc-{slug}"], capture_output=True, text=True)
    out = p.stdout + p.stderr
    for s in ("RUNNING", "COMPLETE", "QUEUED", "ERROR", "CANCEL", "NEW"):
        if s in out:
            return s
    return "UNKNOWN: " + out.strip().splitlines()[-1][:120] if out.strip() else "UNKNOWN"


def push(cfg: dict) -> tuple[bool, str]:
    args = [sys.executable, "-I", str(HERE / "push.py"), "gemma_eval", "--gpu",
            "--kernel-source", "agentloc-b2-dense", "--config", json.dumps(cfg)]
    subprocess.run(args, check=True, capture_output=True)
    src = ROOT / "data" / "kaggle_kernels" / f"agentloc-{cfg['slug']}" / "run.py"
    compile(src.read_text(encoding="utf-8"), str(src), "exec")
    p = subprocess.run(args + ["--push"], capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    return "successfully pushed" in out, out[-200:]


def main() -> None:
    queue, track = list(QUEUE), list(TRACK)
    log(f"start: tracking {track}, queue {[q['slug'] for q in queue]}")
    while True:
        states = {s: status(s) for s in track}
        bad = {s: v for s, v in states.items() if v not in OK}
        if bad:
            log(f"STOP: unexpected status {bad}")
            return
        busy = sum(v in ("RUNNING", "QUEUED") for v in states.values())
        if queue and busy < 2:
            cfg = queue[0]
            ok, out = push(cfg)
            if not ok:
                log(f"STOP: push error for {cfg['slug']}: {out}")
                return
            log(f"pushed {cfg['slug']}")
            track.append(cfg["slug"])
            queue.pop(0)
            time.sleep(60)
            continue
        if not queue and busy == 0:
            log(f"done: all complete {states}")
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
