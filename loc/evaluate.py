"""Score run files against gold, with subsets and repository-clustered bootstrap CIs.

  python -I loc/evaluate.py [--runs data/v0.1/runs] [--out data/v0.1/results]

For every run file (<method>.jsonl) and subset it reports file- and
function-level metrics (Acc@k all-gold, Hit@k any-gold, MRR, NDCG@k) as the
micro mean with a 95% bootstrap CI (1,000 resamples of repositories with
replacement, seed 0), plus the repository-macro mean.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import V01, load, read_jsonl  # noqa: E402
from metrics import report  # noqa: E402

KEYS_FILE = ["acc_all@1", "acc_all@5", "hit@1", "hit@5", "hit@10", "mrr", "ndcg@5"]
KEYS_FUNC = ["acc_all@10", "hit@5", "hit@10", "mrr"]

SUBSETS = {
    "all": lambda r: True,
    "T2": lambda r: r["tier"] == "T2",
    "T1": lambda r: r["tier"] == "T1",
    "T0": lambda r: r["tier"] == "T0",
    "born_after": lambda r: r["born_after_cutoff"],
    "old_repo_T2": lambda r: (not r["born_after_cutoff"]) and r["tier"] == "T2",
    "independent": lambda r: r["independent"],
    "independent_T2": lambda r: r["independent"] and r["tier"] == "T2",
    "no_file_mention": lambda r: not r["flags"]["issue_mentions_gold_file"],
    "not_mostly_new_code": lambda r: not r["flags"].get("mostly_new_code"),
    "openminion": lambda r: r["repo"] == "openminion/openminion",
}


def bootstrap(rows: list[tuple[str, float]], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    by_repo = defaultdict(list)
    for repo, v in rows:
        by_repo[repo].append(v)
    repos = sorted(by_repo)
    mean = sum(v for _, v in rows) / len(rows)
    if len(repos) < 2:
        return mean, float("nan"), float("nan")
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        vals = [v for rp in (rng.choice(repos) for _ in repos) for v in by_repo[rp]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return mean, stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def score(insts: dict[str, dict], run: list[dict]) -> list[dict]:
    out = []
    for p in run:
        r = insts.get(p["instance_id"])
        if not r:
            continue
        row = {"instance_id": r["instance_id"], "repo": r["repo"],
               "file": report(p["files"], set(r["gold_files"]))}
        if "functions" in p and r["gold_functions"]:
            row["func"] = report(p["functions"], set(r["gold_functions"]))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(V01 / "runs"))
    ap.add_argument("--out", default=str(V01 / "results"))
    args = ap.parse_args()
    insts = {r["instance_id"]: r for r in load()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    for f in sorted(Path(args.runs).glob("*.jsonl")):
        method = f.stem
        scored = score(insts, read_jsonl(f))
        table[method] = {}
        for sname, pred in SUBSETS.items():
            rows = [s for s in scored if pred(insts[s["instance_id"]])]
            if not rows:
                continue
            res = {"n": len(rows), "repos": len({s["repo"] for s in rows})}
            for level, keys in (("file", KEYS_FILE), ("func", KEYS_FUNC)):
                lrows = [s for s in rows if level in s]
                for k in keys:
                    if not lrows:
                        continue
                    m, lo, hi = bootstrap([(s["repo"], s[level][k]) for s in lrows])
                    per_repo = defaultdict(list)
                    for s in lrows:
                        per_repo[s["repo"]].append(s[level][k])
                    macro = sum(sum(v) / len(v) for v in per_repo.values()) / len(per_repo)
                    res[f"{level}.{k}"] = {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                                           "macro": round(macro, 4), "n": len(lrows)}
            table[method][sname] = res
    (out / "baselines.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    # compact markdown summary
    lines = ["| method | subset | n | file Hit@1 | file Hit@5 | file Acc@5 | file MRR | func Hit@10 | func Acc@10 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for method, subs in table.items():
        for sname, res in subs.items():
            def g(k):
                v = res.get(k)
                return f"{v['mean']:.3f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]" if v else "-"
            lines.append(f"| {method} | {sname} | {res['n']} | {g('file.hit@1')} | {g('file.hit@5')} | "
                         f"{g('file.acc_all@5')} | {g('file.mrr')} | {g('func.hit@10')} | {g('func.acc_all@10')} |")
    (out / "baselines.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(chr(10).join(lines))


if __name__ == "__main__":
    main()
