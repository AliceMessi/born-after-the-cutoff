"""Print compact audit cards for gold-audit instances (reads only cached API data).

  python -I harvest/audit_view.py --sample 60 --seed 0 --batch 0 --size 10

The sample is stratified: proportional over tiers, at least 10 born-after and at
least 5 OpenMinion instances, otherwise random (seeded). The sample ids are
written to data/v0.1/audit_ids.txt so the audit is reproducible.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harvest"))
sys.path.insert(0, str(ROOT / "loc"))
import gh  # noqa: E402
from data import V01, load  # noqa: E402
from paths import is_core_py  # noqa: E402


def sample(insts: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    om = [r for r in insts if r["repo"] == "openminion/openminion"]
    born = [r for r in insts if r["born_after_cutoff"] and r["repo"] != "openminion/openminion"]
    rest = [r for r in insts if not r["born_after_cutoff"]]
    picked = rng.sample(om, min(5, len(om)))
    picked += rng.sample(born, min(n // 2 - len(picked), len(born)))
    picked += rng.sample(rest, min(n - len(picked), len(rest)))
    rng.shuffle(picked)
    return picked


def card(r: dict, max_patch_lines: int = 45) -> str:
    out = [f"=== {r['instance_id']}  tier={r['tier']} born_after={r['born_after_cutoff']} "
           f"independent={r['independent']} kind={r['source_kind']}",
           f"fix: {r['fix_url']}",
           f"ISSUE: {r['problem_statement'][:900]}",
           f"GOLD files: {r['gold_files']}",
           f"GOLD funcs: {r['gold_functions']}",
           f"added symbols: {r['added_symbols']}  module-level: {r['module_level_change_files']}",
           f"flags: {r['flags']}  same_author={r['same_author']} gap_h={r['issue_to_fix_hours']}"]
    commit = gh.rest(f"repos/{r['repo']}/commits/{r['merge_commit']}")
    out.append(f"COMMIT: {commit['commit']['message'].splitlines()[0][:150]}")
    for f in commit.get("files") or []:
        if not is_core_py(f["filename"]):
            continue
        lines = (f.get("patch") or "").splitlines()
        out.append(f"--- {f['filename']} (+{f['additions']} -{f['deletions']})")
        out += [ln[:160] for ln in lines[:max_patch_lines]]
        if len(lines) > max_patch_lines:
            out.append(f"... ({len(lines) - max_patch_lines} more lines)")
    return "\n".join(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--size", type=int, default=10)
    args = ap.parse_args()
    ids_file = V01 / "audit_ids.txt"
    insts = {r["instance_id"]: r for r in load()}
    if ids_file.exists():
        ids = [x for x in ids_file.read_text(encoding="utf-8").split() if x in insts]
    else:
        ids = [r["instance_id"] for r in sample(list(insts.values()), args.sample, args.seed)]
        ids_file.write_text(chr(10).join(ids) + chr(10), encoding="utf-8")
    for iid in ids[args.batch * args.size : (args.batch + 1) * args.size]:
        print(card(insts[iid]))
        print()


if __name__ == "__main__":
    main()
