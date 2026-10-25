"""Select the step-3 pilot ids (PREREG §6): 30 T1 instances outside the evaluation
set, balanced over repositories (at most 3 each), half from born-after repositories,
with the evaluation exclusions applied. Used only to fix prompts and parsing.

  python -I harvest/select_pilot.py --seed 0   -> data/v0.1/pilot_ids.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harvest"))
sys.path.insert(0, str(ROOT / "loc"))
from data import V01, load  # noqa: E402
from select_eval import take_balanced  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    rng = random.Random(f"pilot:{args.seed}")
    eval_ids = set((V01 / "eval_ids.txt").read_text(encoding="utf-8").split())
    excl = set((V01 / "audit_exclusions.txt").read_text(encoding="utf-8").split())
    pool = [r for r in load() if r["tier"] == "T1" and r["gold_functions"] and r["instance_id"] not in eval_ids
            and r["instance_id"] not in excl and r["repo"] != "openminion/openminion"
            and not (r["flags"].get("mostly_new_code") or r["flags"].get("issue_older_than_1y"))]
    born = take_balanced([r for r in pool if r["born_after_cutoff"]], args.n // 2, 3, rng)
    old = take_balanced([r for r in pool if not r["born_after_cutoff"]], args.n - len(born), 3, rng)
    ids = [r["instance_id"] for r in born + old]
    (V01 / "pilot_ids.txt").write_text(chr(10).join(ids) + chr(10), encoding="utf-8")
    print(len(ids), "pilot ids;", len({r["repo"] for r in born + old}), "repos;", len(born), "born-after")


if __name__ == "__main__":
    main()
