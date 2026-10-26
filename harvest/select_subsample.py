"""Seeded subsample of an id list, balanced between born-after and old repositories.

  python -I harvest/select_subsample.py --from data/v0.1/g2_ids.txt --n 150 --seed 0 --out data/v0.1/ids_26b.txt

Used for the optional 26B-A4B run: 150 of the 200 G2-subsample ids (75 born-after, 75 old),
so P, A and G1 with 26B-A4B are paired with every other method on the same instances.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loc"))
from data import load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    born = {r["instance_id"]: r["born_after_cutoff"] for r in load()}
    ids = sorted(Path(args.src).read_text(encoding="utf-8").split())
    rng = random.Random(args.seed)
    picked = []
    for flag in (True, False):
        group = [i for i in ids if born[i] is flag]
        picked += rng.sample(group, min(len(group), args.n // 2))
    Path(args.out).write_text(chr(10).join(sorted(picked)) + chr(10), encoding="utf-8")
    print(f"{args.out}: {len(picked)} ids ({sum(born[i] for i in picked)} born-after)")


if __name__ == "__main__":
    main()
