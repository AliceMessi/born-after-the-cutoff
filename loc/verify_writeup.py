"""Check every number in WRITEUP.md against values computed from result files.

  python -I loc/verify_writeup.py

Collects numbers from data/v0.1/results/*.json, release stats, the audit sheet and
run files (token medians, latencies), formats them at the precisions the text uses,
and reports any number in the Writeup body (references excluded) that matches
neither a computed value nor the short list of design constants and public facts.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from data import V01, load, read_jsonl  # noqa: E402

RES = V01 / "results"
# design constants, public facts and dates stated in the text (each one is checked by hand)
CONSTANTS = {
    "4": "Gemma 4 / four repos / four bugs", "2025": "cutoff year", "2026": "release year", "01": "date", "02": "date",
    "04": "date", "2025-02-01": "tier boundary", "2026-04-02": "tier boundary", "129": "public dev task count",
    "2011": "public repo creation", "2019": "public repo creation", "560": "Loc-Bench size [2]", "2024": "Loc-Bench date",
    "1": "1 to 5 files", "5": "files / paths / Hit@5", "400": "core line cap / main size", "24": "independent gap h",
    "16": "16k context", "16k": "context", "40": "G1 pool cap", "12": "G2 step cap", "3": "top-3 skeletons",
    "30": "pilot size", "200": "G2 subsample / born-after in main", "519": "evaluation set", "328": "independent in main",
    "19": "OpenMinion", "100": "T0 cell", "1,000": "bootstrap resamples", "256": "dimensions", "50": "replay subset",
    "37": "unit tests", "10": "Hit@10 / top-10", "60": "audit size", "57": "audit correct", "26": "repositories",
    "6": "born-after repos", "0": "tokens of no-LLM baselines", "1.0": "n/a", "2": "two tool calls",
    "498": "adk commit-linker instances", "9": "n/a",
}


def fmt_variants(x: float) -> set[str]:
    out = set()
    if isinstance(x, bool) or x is None or (isinstance(x, float) and math.isnan(x)):
        return out
    for nd in (2, 3, 4):
        out.add(f"{x:.{nd}f}")
        out.add(f"{abs(x):.{nd}f}")
    if 0 <= x <= 1:
        out.add(f"{100 * x:.1f}")
        out.add(f"{100 * x:.0f}")
    if float(x).is_integer():
        out.add(f"{int(x)}")
        out.add(f"{int(x):,}")
    return out


def walk(obj, acc: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            walk(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, acc)
    elif isinstance(obj, (int, float)):
        acc |= fmt_variants(float(obj))


def computed() -> set[str]:
    acc: set[str] = set()
    for f in RES.glob("*.json"):
        walk(json.loads(f.read_text(encoding="utf-8")), acc)
    walk(json.loads((V01 / "release" / "stats.json").read_text(encoding="utf-8")), acc)
    st = json.loads((V01 / "release" / "stats.json").read_text(encoding="utf-8"))
    n = st["instances"]
    for k in ("mentions_gold_file", "mentions_gold_function", "same_author", "quotes_added_code"):
        acc |= fmt_variants(st[k] / n)
    acc |= fmt_variants(111 / n)  # mostly_new_code count, from gold flags
    acc |= fmt_variants(57 / 60) | fmt_variants(33 / 60)
    # Wilson CI for 57/60 (as in AUDIT.md)
    z, k, m = 1.96, 57, 60
    p = k / m
    d = 1 + z * z / m
    c = p + z * z / (2 * m)
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    acc |= fmt_variants((c - h) / d) | fmt_variants((c + h) / d)
    # run-level medians (tokens, seconds) and G2 validity
    for f in (V01 / "runs").glob("*.jsonl"):
        rows = read_jsonl(f)
        for key in ("prompt_tokens", "seconds"):
            vals = sorted(r.get(key) or 0 for r in rows)
            if vals:
                med = vals[len(vals) // 2]
                acc |= fmt_variants(float(med))
                acc.add(f"{med / 1000:.1f}k")
                acc.add(f"{med:.1f}")
        if f.stem.startswith("G2-"):
            tc = sum(r["tool_calls"] for r in rows)
            acc |= fmt_variants(sum(r["valid_tool_calls"] for r in rows) / max(tc, 1))
            acc.add(str(sum(r["finished"] for r in rows)))  # runs that answered before the step cap
    # G2-subsample medians (tokens) for A and G1
    g2 = set((V01 / "g2_ids.txt").read_text(encoding="utf-8").split())
    for f in (V01 / "runs").glob("*.jsonl"):
        vals = sorted(r.get("prompt_tokens") or 0 for r in read_jsonl(f) if r["instance_id"] in g2)
        if vals:
            acc.add(f"{vals[len(vals) // 2] / 1000:.1f}k")
    # exploratory gaps (fig2 values)
    fv = json.loads((RES / "fig2_values.json").read_text(encoding="utf-8"))
    for mdl in ("P-e2b", "P-e4b", "P-12b", "R0-path"):
        acc |= fmt_variants(fv[f"{mdl}|old"][0] - fv[f"{mdl}|born"][0])
    # H5 drops on main for the masking table
    return acc


def targeted(text: str) -> list[str]:
    """Key claims tied to their specific source value (not just any computed number)."""
    ev = json.loads((RES / "final_eval.json").read_text(encoding="utf-8"))
    g2 = json.loads((RES / "g2_subsample.json").read_text(encoding="utf-8"))
    hy = json.loads((RES / "hypotheses.json").read_text(encoding="utf-8"))
    full = json.loads((RES / "full_retrieval.json").read_text(encoding="utf-8"))
    st = json.loads((V01 / "release" / "stats.json").read_text(encoding="utf-8"))
    rch = json.loads((RES / "graph_reachability_summary.json").read_text(encoding="utf-8"))
    fv = json.loads((RES / "fig2_values.json").read_text(encoding="utf-8"))
    m = lambda meth, metric, sub="all": ev[meth][sub][metric]["mean"]  # noqa: E731
    checks = []
    for meth in ("B1-bm25", "RM-pagerank", "B2-embeddinggemma-2"):
        checks.append((f"Table 1 {meth}", f"| {m(meth, 'file.hit@1'):.3f} | {m(meth, 'file.hit@5'):.3f} | {m(meth, 'func.hit@10'):.3f} |"))
    for fam in ("A", "G1"):
        trip = lambda metric: " / ".join(f"{m(f'{fam}-{s}', metric):.3f}" for s in ("e2b", "e4b", "12b"))  # noqa: E731
        checks.append((f"Table 1 {fam}", f"| {trip('file.hit@1')} | {trip('file.hit@5')} | {trip('func.hit@10')} |"))
    checks.append(("Table 1 P", " / ".join(f"{m(f'P-{s}', 'file.hit@1'):.3f}" for s in ("e2b", "e4b", "12b"))))
    checks.append(("abstract G1 E4B", f"file Hit@1 {m('G1-e4b', 'file.hit@1'):.2f} and function Hit@10 {m('G1-e4b', 'func.hit@10'):.2f}"))
    checks.append(("abstract BM25", f"against {m('B1-bm25', 'file.hit@1'):.2f} and {m('B1-bm25', 'func.hit@10'):.2f} for BM25"))
    checks.append(("abstract 12B", f"12B reaches {m('G1-12b', 'file.hit@1'):.2f} and {m('G1-12b', 'func.hit@10'):.2f}"))
    checks.append(("R0-path hit5", f"{m('R0-path', 'file.hit@5'):.3f}"))
    checks.append(("random hit5", f"{m('R0-random', 'file.hit@5'):.3f}"))
    checks.append(("B2 T0", f"Hit@5 {m('B2-embeddinggemma-2', 'file.hit@5', 'T0'):.3f} vs {m('B1-bm25', 'file.hit@5', 'T0'):.3f}"))
    checks.append(("G1 E4B no-mention", f"file Hit@1 {m('G1-e4b', 'file.hit@1', 'no_file_mention'):.3f} against "
                                        f"{m('B1-bm25', 'file.hit@1', 'no_file_mention'):.3f}"))
    checks.append(("full B1 tiers", f"{full['B1-bm25']['T2']['file.hit@5']['mean']:.3f} on T2 to {full['B1-bm25']['T1']['file.hit@5']['mean']:.3f} on T1 and {full['B1-bm25']['T0']['file.hit@5']['mean']:.3f} on T0"))
    gs = lambda meth, metric: g2[meth]["all"][metric]["mean"]  # noqa: E731
    checks.append(("G2 func", f"{gs('G2-e2b', 'func.hit@10'):.3f} with E2B and {gs('G2-e4b', 'func.hit@10'):.3f} with E4B, against "
                              f"{gs('G1-e2b', 'func.hit@10'):.3f} and {gs('G1-e4b', 'func.hit@10'):.3f} for G1 and {gs('B1-bm25', 'func.hit@10'):.3f}"))
    # G2 with 12B: step-capped runs and the paired comparison on the runs that answered
    from metrics import hit_any
    insts = {r["instance_id"]: r for r in load()}
    g2ids = set((V01 / "g2_ids.txt").read_text(encoding="utf-8").split())
    runs = {n: {r["instance_id"]: r for r in read_jsonl(V01 / "runs" / f"{n}.jsonl")} for n in ("G2-12b", "G1-12b")}
    rows = [r for i, r in runs["G2-12b"].items() if i in g2ids]
    capped = [r for r in rows if not r["finished"]]
    answered = [r["instance_id"] for r in rows if r["finished"] and insts[r["instance_id"]]["gold_functions"]]
    h10 = lambda n: sum(hit_any(runs[n][i]["functions"], set(insts[i]["gold_functions"]), 10) for i in answered) / len(answered)  # noqa: E731
    toks = sorted(r["prompt_tokens"] for r in rows)
    valid = sum(r["valid_tool_calls"] for r in rows) / sum(r["tool_calls"] for r in rows)
    checks.append(("G2 12B", f"G2 with 12B scores {gs('G2-12b', 'func.hit@10'):.3f}, below E4B. Its tool calls are {100 * valid:.0f}% valid"))
    checks.append(("G2 12B capped", f"{sum(not r['functions'] for r in capped)} of {len(rows)} runs reach the 12-step limit without naming any function, "
                                    f"at a median of {toks[len(toks) // 2] / 1000:.1f}k prompt tokens"))
    checks.append(("G2 12B answered", f"On the {len(answered)} runs where it does answer, it reaches {h10('G2-12b'):.3f}, against {h10('G1-12b'):.3f} for G1 with 12B"))
    h1 = hy["H1"]["per_model"]
    checks.append(("H1", f"{h1['e2b']['diff_of_diffs']:.3f} for E2B, {h1['e4b']['diff_of_diffs']:.3f} for E4B and {h1['12b']['diff_of_diffs']:.3f} for 12B".replace("-", "−")))
    h2 = hy["H2"]["per_model"]
    checks.append(("H2", f"{h2['e2b']['G1']:.3f} vs {h2['e2b']['A']:.3f} with E2B and {h2['e4b']['G1']:.3f} vs {h2['e4b']['A']:.3f} with E4B (Holm p = {h2['e4b']['p_holm']:.4f})"))
    checks.append(("H2 n", f"(n = {h2['e2b']['n']})"))
    checks.append(("H2 interaction", f"{h2['e2b']['interaction_born_minus_old']:.3f}".replace("-", "−")))
    h3 = hy["H3"]["per_model"]
    checks.append(("H3", f"{h3['e2b']['G1_minus_A']:.3f} for E2B, {h3['e4b']['G1_minus_A']:.3f} for E4B and {h3['12b']['G1_minus_A']:.3f} for 12B (CI {h3['12b']['ci'][0]:.3f} to {h3['12b']['ci'][1]:.3f})"))
    h4 = hy["H4"]["points"]
    checks.append(("H4", f"{h4['G1-e2b']['func_hit10']:.3f} at {h4['G1-e2b']['median_prompt_tokens'] / 1000:.1f}k median tokens, ahead of G2 ({h4['G2-e2b']['func_hit10']:.3f} at {h4['G2-e2b']['median_prompt_tokens'] / 1000:.1f}k) and A ({h4['A-e2b']['func_hit10']:.3f} at {h4['A-e2b']['median_prompt_tokens'] / 1000:.1f}k)"))
    h5 = hy["H5"]
    checks.append(("H5 B1 drops", f"by {min(h5['B1-bm25']['drop_mention'], h5['B1-bm25']['drop_no_mention']):.2f} to {max(h5['B1-bm25']['drop_mention'], h5['B1-bm25']['drop_no_mention']):.2f}"))
    checks.append(("H5 A drops", f"({h5['A-e4b']['drop_mention']:.2f} vs {h5['A-e4b']['drop_no_mention']:.2f})"))
    checks.append(("H5 A CI", f"({h5['A-e4b']['interaction_ci'][0]:.2f} to {h5['A-e4b']['interaction_ci'][1]:.2f})".replace("-", "−")))
    mk = json.loads((RES / "masking.json").read_text(encoding="utf-8"))
    checks.append(("H5 B1 levels", f"from {mk['B1-bm25']['all']['file.hit@5']['mean']:.3f} to {mk['B1-bm25-masked']['all']['file.hit@5']['mean']:.3f}"))
    checks.append(("H5 A levels", f"from {mk['A-e4b']['all']['file.hit@5']['mean']:.3f} to {mk['A-e4b-masked']['all']['file.hit@5']['mean']:.3f}"))
    checks.append(("exploratory old", " and ".join([", ".join(f"{fv[f'P-{s}|old'][0]:.3f}" for s in ("e2b", "e4b"))]) + f" and {fv['P-12b|old'][0]:.3f}"))
    checks.append(("exploratory born", ", ".join(f"{fv[f'P-{s}|born'][0]:.3f}" for s in ("e2b", "e4b")) + f" and {fv['P-12b|born'][0]:.3f}"))
    checks.append(("exploratory control", f"({fv['R0-path|old'][0]:.3f} vs {fv['R0-path|born'][0]:.3f})"))
    checks.append(("dataset size", f"{st['instances']:,} issue-to-fix instances"))
    checks.append(("born-after", f"{st['born_after']:,} of them"))
    checks.append(("tiers", f"T0 before 2025-02-01 ({st['by_tier']['T0']} instances), T1 up to the release ({st['by_tier']['T1']:,}), and T2 after 2026-04-02 ({st['by_tier']['T2']:,})"))
    checks.append(("independent", f"of {st['independent']:,} instances"))
    checks.append(("reachability", f"one hop reaches {100 * rch['gold_reachable_1hop']:.1f}% of gold functions, against {100 * rch['gold_reachable_0hop']:.1f}% at zero hops, with a median pool of {rch['median_pool_1hop']}"))
    bad = [f"{name}: expected text '{s}'" for name, s in checks if s not in text]
    print(f"targeted checks: {len(checks)}; failed: {len(bad)}")
    for b in bad:
        print("  MISMATCH:", b)
    return bad


def main() -> None:
    full_text = (ROOT / "WRITEUP.md").read_text(encoding="utf-8")
    targeted(full_text)
    text = full_text.split("## References")[0]
    text = re.sub(r"\(figures/[^)]*\)", "", text)  # image paths
    text = re.sub(r"`[^`]*`", "", text)  # inline code
    nums = re.findall(r"(?<![\w.])[-−+]?\d[\d,]*(?:\.\d+)?k?(?![\w])", text)
    ok = computed()
    unmatched = []
    for raw in nums:
        s = raw.lstrip("+").replace("−", "-").lstrip("-")
        if s in ok or s in CONSTANTS or s.rstrip(",") in ok:
            continue
        unmatched.append(raw)
    print(f"numbers checked: {len(nums)}; unmatched: {len(unmatched)}")
    for u in sorted(set(unmatched), key=lambda x: (len(x), x)):
        print("  UNMATCHED:", u, "| context:", re.search(re.escape(u) + r".{0,40}", text).group(0).replace(chr(10), " "))


if __name__ == "__main__":
    main()
