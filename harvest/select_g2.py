"""Seeded G2 subsample (compute budget, PREREG §8): 200 ids from the `main` stratum,
100 born-after + 100 old-repository, independent reports preferred, at most 20 per repo.

  python -I harvest/select_g2.py --seed 0   -> data/v0.1/g2_ids.txt
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(f"g2:{args.seed}")
    eval_ids = (V01 / "eval_ids.txt").read_text(encoding="utf-8").split()
    manifest = json.loads((V01 / "eval_manifest.json").read_text(encoding="utf-8"))
    main_n = manifest["strata"]["main"]["n"]
    main_ids = set(eval_ids[:main_n])  # select_eval writes the `main` stratum first
    pool = [r for r in load() if r["instance_id"] in main_ids and r["gold_functions"]]
    born = take_balanced([r for r in pool if r["born_after_cutoff"]], args.n // 2, 20, rng, prefer=lambda r: r["independent"])
    old = take_balanced([r for r in pool if not r["born_after_cutoff"]], args.n - len(born), 20, rng,
                        prefer=lambda r: r["independent"])
    ids = [r["instance_id"] for r in born + old]
    (V01 / "g2_ids.txt").write_text(chr(10).join(ids) + chr(10), encoding="utf-8")
    print(len(ids), "G2 ids;", len(born), "born-after;", sum(r["independent"] for r in born + old), "independent;",
          len({r["repo"] for r in born + old}), "repos")


if __name__ == "__main__":
    main()
