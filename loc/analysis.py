"""Pre-registered analyses H1-H5 (PREREG.md §4), computed from run files.

  python -I loc/analysis.py            -> data/v0.1/results/hypotheses.{json,md}

Conventions (fixed in PREREG): file Hit@5 and function Hit@10 per instance; 1,000
repository-clustered bootstrap resamples (seed 0) for CIs; paired sign-flip
permutation tests with 10,000 permutations (seed 0); Holm correction across the
two models in H2. Primary subset for H2 is independent ∩ main (n >= 150).
A hypothesis whose run files are missing is reported as "not run".
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import V01, load, read_jsonl  # noqa: E402
from metrics import hit_any  # noqa: E402

RUNS = V01 / "runs"
B = 1000
PERM = 10000


def runs(name: str) -> dict[str, dict] | None:
    f = RUNS / f"{name}.jsonl"
    return {r["instance_id"]: r for r in read_jsonl(f)} if f.exists() else None


def fh5(run, r):
    return hit_any(run[r["instance_id"]].get("files", []), set(r["gold_files"]), 5)


def fn10(run, r):
    return hit_any(run[r["instance_id"]].get("functions", []), set(r["gold_functions"]), 10)


def cluster_boot(groups: dict[str, list[tuple[str, float]]], stat, seed: int = 0) -> tuple[float, float, float]:
    """groups: name -> [(repo, value)]; stat: f({name: [values]}) -> float. Resample repos within groups."""
    by = {g: defaultdict(list) for g in groups}
    for g, rows in groups.items():
        for repo, v in rows:
            by[g][repo].append(v)
    point = stat({g: [v for _, v in rows] for g, rows in groups.items()})
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        sample = {}
        for g, repos in by.items():
            keys = sorted(repos)
            sample[g] = [v for k in (rng.choice(keys) for _ in keys) for v in repos[k]]
        vals.append(stat(sample))
    vals.sort()
    return point, vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def sign_flip(diffs: list[float], seed: int = 0) -> float:
    obs = abs(sum(diffs))
    rng = random.Random(seed)
    hits = sum(1 for _ in range(PERM) if abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12)
    return (hits + 1) / (PERM + 1)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    insts = {r["instance_id"]: r for r in load()}
    ids = (V01 / "eval_ids.txt").read_text(encoding="utf-8").split()
    man = json.loads((V01 / "eval_manifest.json").read_text(encoding="utf-8"))
    n_main = man["strata"]["main"]["n"]
    main_ids = [i for i in ids[:n_main] if i in insts]
    t0_ids = [i for i in ids if i in insts and insts[i]["tier"] == "T0" and not insts[i]["born_after_cutoff"]]
    g2_ids = [i for i in (V01 / "g2_ids.txt").read_text(encoding="utf-8").split() if i in insts]
    out: dict = {}
    r0 = runs("R0-path")

    # H1: memory vs skill
    h1 = {}
    for m in ("e2b", "e4b", "12b"):
        p = runs(f"P-{m}")
        if not (p and r0):
            h1[m] = "not run"
            continue
        groups = {
            "t0_old": [(insts[i]["repo"], fh5(p, insts[i]) - fh5(r0, insts[i])) for i in t0_ids if i in p and i in r0],
            "born_after": [(insts[i]["repo"], fh5(p, insts[i]) - fh5(r0, insts[i]))
                           for i in main_ids if insts[i]["born_after_cutoff"] and i in p and i in r0],
        }
        pt, lo, hi = cluster_boot(groups, lambda s: mean(s["t0_old"]) - mean(s["born_after"]))
        h1[m] = {"delta_t0_old": round(mean([v for _, v in groups["t0_old"]]), 4),
                 "delta_born_after": round(mean([v for _, v in groups["born_after"]]), 4),
                 "diff_of_diffs": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)],
                 "n": [len(groups["t0_old"]), len(groups["born_after"])], "ci_excludes_0": lo > 0 or hi < 0}
    supported = [m for m, v in h1.items() if isinstance(v, dict) and v["ci"][0] > 0]
    out["H1"] = {"per_model": h1, "supported_models": supported,
                 "verdict": "supported" if len(supported) >= 2 else ("not run" if all(v == "not run" for v in h1.values())
                                                                     else "not supported")}

    # H2: G1 vs A, function Hit@10, primary subset independent ∩ main; interaction born-after vs old
    h2, pvals = {}, {}
    primary = [i for i in main_ids if insts[i]["independent"] and insts[i]["gold_functions"]]
    if len(primary) < 150:
        primary = [i for i in main_ids if insts[i]["gold_functions"]]
    for m in ("e2b", "e4b"):
        g1, a = runs(f"G1-{m}"), runs(f"A-{m}")
        if not (g1 and a):
            h2[m] = "not run"
            continue
        sub = [i for i in primary if i in g1 and i in a]
        d = [fn10(g1, insts[i]) - fn10(a, insts[i]) for i in sub]
        pvals[m] = sign_flip(d)
        groups = {"born": [(insts[i]["repo"], fn10(g1, insts[i]) - fn10(a, insts[i])) for i in sub if insts[i]["born_after_cutoff"]],
                  "old": [(insts[i]["repo"], fn10(g1, insts[i]) - fn10(a, insts[i])) for i in sub if not insts[i]["born_after_cutoff"]]}
        pt, lo, hi = cluster_boot(groups, lambda s: mean(s["born"]) - mean(s["old"]))
        h2[m] = {"n": len(sub), "G1": round(mean([fn10(g1, insts[i]) for i in sub]), 4),
                 "A": round(mean([fn10(a, insts[i]) for i in sub]), 4), "mean_diff": round(mean(d), 4),
                 "p_perm": round(pvals[m], 5), "interaction_born_minus_old": round(pt, 4),
                 "interaction_ci": [round(lo, 4), round(hi, 4)]}
    ordered = sorted(pvals, key=pvals.get)  # Holm over the models that ran
    for k, m in enumerate(ordered):
        h2[m]["p_holm"] = round(min(1.0, max(pvals[x] * (len(ordered) - j) for j, x in enumerate(ordered[: k + 1]))), 5)
    out["H2"] = {"primary_subset": "independent and main" if len(primary) >= 150 else "main", "per_model": h2}

    # H3: G1 - A by size (descriptive)
    h3 = {}
    for m in ("e2b", "e4b", "12b"):
        g1, a = runs(f"G1-{m}"), runs(f"A-{m}")
        if g1 and a:
            sub = [i for i in main_ids if i in g1 and i in a and insts[i]["gold_functions"]]
            rows = {"x": [(insts[i]["repo"], fn10(g1, insts[i]) - fn10(a, insts[i])) for i in sub]}
            pt, lo, hi = cluster_boot(rows, lambda s: mean(s["x"]))
            h3[m] = {"n": len(sub), "G1_minus_A": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)]}
        else:
            h3[m] = "not run"
    vals = [h3[m]["G1_minus_A"] for m in ("e2b", "e4b", "12b") if isinstance(h3[m], dict)]
    out["H3"] = {"per_model": h3, "monotonic_decrease": len(vals) == 3 and vals[0] > vals[1] > vals[2]}

    # H4: E2B Pareto on the G2 subsample (function Hit@10 vs median prompt tokens)
    h4 = {}
    b1 = runs("B1-bm25")
    for name in ("B1-bm25", "RM-pagerank", "A-e2b", "G1-e2b", "G2-e2b"):
        rr = runs(name)
        if not rr:
            h4[name] = "not run"
            continue
        sub = [i for i in g2_ids if i in rr]
        toks = sorted(rr[i].get("prompt_tokens") or 0 for i in sub)
        h4[name] = {"n": len(sub), "func_hit10": round(mean([fn10(rr, insts[i]) for i in sub]), 4),
                    "median_prompt_tokens": toks[len(toks) // 2] if toks else None}
    if isinstance(h4.get("G1-e2b"), dict) and isinstance(h4.get("G2-e2b"), dict):
        g1, g2 = h4["G1-e2b"], h4["G2-e2b"]
        out["H4"] = {"points": h4, "G1_dominates_G2": g1["func_hit10"] >= g2["func_hit10"]
                     and g1["median_prompt_tokens"] < g2["median_prompt_tokens"]}
    else:
        out["H4"] = {"points": h4, "G1_dominates_G2": "not run"}

    # H5: identifier masking, file Hit@5, mention vs no-mention
    h5 = {}
    for orig, masked in (("B1-bm25", "B1-bm25-masked"), ("A-e4b", "A-e4b-masked")):
        o, k = runs(orig), runs(masked)
        if not (o and k):
            h5[orig] = "not run"
            continue
        sub = [i for i in main_ids if i in o and i in k]
        groups = {"mention": [(insts[i]["repo"], fh5(o, insts[i]) - fh5(k, insts[i])) for i in sub
                              if insts[i]["flags"]["issue_mentions_gold_file"]],
                  "no_mention": [(insts[i]["repo"], fh5(o, insts[i]) - fh5(k, insts[i])) for i in sub
                                 if not insts[i]["flags"]["issue_mentions_gold_file"]]}
        pt, lo, hi = cluster_boot(groups, lambda s: mean(s["mention"]) - mean(s["no_mention"]))
        h5[orig] = {"drop_mention": round(mean([v for _, v in groups["mention"]]), 4),
                    "drop_no_mention": round(mean([v for _, v in groups["no_mention"]]), 4),
                    "p_mention": round(sign_flip([v for _, v in groups["mention"]]), 5),
                    "p_no_mention": round(sign_flip([v for _, v in groups["no_mention"]]), 5),
                    "interaction": round(pt, 4), "interaction_ci": [round(lo, 4), round(hi, 4)],
                    "n": [len(groups["mention"]), len(groups["no_mention"])]}
    out["H5"] = h5
    res = V01 / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "hypotheses.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
