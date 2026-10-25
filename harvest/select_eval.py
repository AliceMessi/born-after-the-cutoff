"""Select the Gemma evaluation set from the v0.1 instances (seeded, documented).

Strata (see PREREG.md):
  main      T2 instances, at most 60 per repository, half from born-after repositories
            when available, and independent reports preferred (>= 60% when possible); target 400
  om        every OpenMinion curated instance with file-level gold (case study)
  rq2_t0    T0 instances from old repositories (incl. the competition's repos); target 100
  rq2_old   T2 instances from old repositories not already in `main`, topped up to 100 in total

  python -I harvest/select_eval.py --seed 0
Writes data/v0.1/eval_ids.txt and data/v0.1/eval_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loc"))
from data import V01, load  # noqa: E402


def take_balanced(pool: list[dict], n: int, cap: int, rng: random.Random, prefer=lambda r: False) -> list[dict]:
    """Round-robin over repositories (shuffled), preferred items first, at most `cap` per repo."""
    by_repo = defaultdict(list)
    for r in pool:
        by_repo[r["repo"]].append(r)
    for rows in by_repo.values():
        rng.shuffle(rows)
        rows.sort(key=lambda r: not prefer(r))  # stable: preferred first, random within groups
    repos = sorted(by_repo)
    rng.shuffle(repos)
    out, taken = [], Counter()
    while len(out) < n:
        progressed = False
        for rp in repos:
            if len(out) >= n:
                break
            if taken[rp] < min(cap, len(by_repo[rp])):
                out.append(by_repo[rp][taken[rp]])
                taken[rp] += 1
                progressed = True
        if not progressed:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--main", type=int, default=400)
    ap.add_argument("--cap", type=int, default=60)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    insts = [r for r in load() if r["gold_files"]]
    excluded = set()
    excl_file = V01 / "audit_exclusions.txt"  # instances the gold audit judged wrong (AUDIT.md)
    if excl_file.exists():
        excluded = {x.strip() for x in excl_file.read_text(encoding="utf-8").splitlines() if x.strip()}
    # mostly-new-code fixes (modified files often incidental) leave the evaluation strata
    insts = [r for r in insts if r["instance_id"] not in excluded
             and (r["repo"] == "openminion/openminion"
                  or not (r["flags"].get("mostly_new_code") or r["flags"].get("issue_older_than_1y")))]
    om = [r for r in insts if r["repo"] == "openminion/openminion"]  # case-study slice, kept whole
    rest = [r for r in insts if r["repo"] != "openminion/openminion"]
    t2 = [r for r in rest if r["tier"] == "T2"]
    born = take_balanced([r for r in t2 if r["born_after_cutoff"]], args.main // 2, args.cap, rng,
                         prefer=lambda r: r["independent"])
    old = take_balanced([r for r in t2 if not r["born_after_cutoff"]], args.main - len(born), args.cap, rng,
                        prefer=lambda r: r["independent"])
    main_set = born + old
    if len(main_set) < args.main:  # top up from whichever side has more left
        used = {r["instance_id"] for r in main_set}
        main_set += take_balanced([r for r in t2 if r["instance_id"] not in used], args.main - len(main_set),
                                  args.cap, rng, prefer=lambda r: r["independent"])
    used = {r["instance_id"] for r in main_set}
    rq2_t0 = take_balanced([r for r in rest if r["tier"] == "T0" and not r["born_after_cutoff"]], 100, 25, rng)
    old_in_main = sum(1 for r in main_set if not r["born_after_cutoff"])
    rq2_old = take_balanced([r for r in t2 if not r["born_after_cutoff"] and r["instance_id"] not in used],
                            max(0, 100 - old_in_main), 25, rng)
    strata = {"main": main_set, "om": om, "rq2_t0": rq2_t0, "rq2_old": rq2_old}
    ids, manifest = [], {"seed": args.seed, "strata": {}}
    for name, rows in strata.items():
        manifest["strata"][name] = {
            "n": len(rows),
            "born_after": sum(r["born_after_cutoff"] for r in rows),
            "independent": sum(r["independent"] for r in rows),
            "by_repo": dict(Counter(r["repo"] for r in rows).most_common()),
            "by_tier": dict(Counter(r["tier"] for r in rows)),
        }
        ids += [r["instance_id"] for r in rows]
    ids = list(dict.fromkeys(ids))
    (V01 / "eval_ids.txt").write_text(chr(10).join(ids) + chr(10), encoding="utf-8")
    (V01 / "eval_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "born_after", "independent", "by_tier")}
                      for k, v in manifest["strata"].items()}, indent=1))
    print("total ids:", len(ids))


if __name__ == "__main__":
    main()
