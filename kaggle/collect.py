"""Download run files from finished private kernels into data/v0.1/runs.

  python -I kaggle/collect.py gemma-e2b gemma-e4b ...    (slugs without the agentloc- prefix)

Skips kernels that are not COMPLETE; never downloads embedding caches or logs into runs/.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "kaggle_out"
RUNS = ROOT / "data" / "v0.1" / "runs"


def main() -> None:
    for slug in sys.argv[1:]:
        st = subprocess.run(["kaggle", "kernels", "status", f"<kaggle-user>/agentloc-{slug}"], capture_output=True, text=True)
        status = (st.stdout + st.stderr).strip()
        if "COMPLETE" not in status:
            print(f"{slug}: {status.split('status')[-1].strip()}")
            continue
        dest = OUT / slug
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        subprocess.run(["kaggle", "kernels", "output", f"<kaggle-user>/agentloc-{slug}", "-p", str(dest)],
                       capture_output=True, text=True)
        for big in list(dest.rglob("*.sqlite")) + list(dest.rglob("ollama.log")):
            big.unlink()
        got = []
        for f in (dest / "out").glob("*.jsonl"):
            shutil.copy(f, RUNS / f.name)
            got.append(f"{f.name}({sum(1 for _ in f.open(encoding='utf-8'))})")
        print(f"{slug}: COMPLETE -> {', '.join(got) or 'no run files'}")


if __name__ == "__main__":
    main()
