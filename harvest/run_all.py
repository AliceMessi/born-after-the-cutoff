"""Run every harvest in a manifest, one at a time (low RAM, rate-limit friendly).

  python -I harvest/run_all.py harvest/repos_v01.json --out data/v0.1/raw [--force]

Skips runs whose funnel file already exists unless --force. Each run is a
separate `python -I harvest/harvest.py` process; failures are logged and the
loop continues.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="data/v0.1/raw")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on repo name")
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out)
    for run in man["runs"]:
        if args.only and args.only not in run["repo"]:
            continue
        stem = f"{run['repo'].replace('/', '__')}.{run['mode']}" + (f".{run['tag']}" if run.get("tag") else "")
        if (out / f"{stem}.funnel.json").exists() and not args.force:
            print(f"skip {stem} (done)", flush=True)
            continue
        cmd = [sys.executable, "-I", str(HERE / "harvest.py"), "--repo", run["repo"], "--mode", run["mode"],
               "--until", run.get("until", man["until"]), "--out", str(out)]
        if run.get("since"):
            cmd += ["--since", run["since"]]
        if run.get("pairs"):
            cmd += ["--pairs", run["pairs"]]
        if run.get("tag"):
            cmd += ["--tag", run["tag"]]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
        err = [ln for ln in (p.stderr or "").splitlines() if "[rate]" in ln or "Error" in ln or "Traceback" in ln]
        print(f"{'OK ' if p.returncode == 0 else 'ERR'} {stem}: {tail[0]} {' | '.join(err[-3:])}", flush=True)


if __name__ == "__main__":
    main()
