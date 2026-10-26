"""Render the Writeup figures as PNG from result files (no hand-typed numbers).

  python -I loc/figures.py   -> figures/fig1_results.png, figures/fig2_familiarity.png

Palette: validated categorical slots 1-2 of the reference palette (blue #2a78d6,
orange #eb6834; adjacent CVD dE 24.7, normal-vision dE 33.6). Identity is never
color-alone: series also differ by marker shape (Fig. 1) and are directly labelled
or carried by a legend (Fig. 2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from analysis import cluster_boot, mean  # noqa: E402
from data import V01, load, read_jsonl  # noqa: E402
from metrics import hit_any  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OUT = ROOT / "figures"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

ROWS = [("G1-12b", "G1 graph rerank · 12B"), ("G1-e4b", "G1 graph rerank · E4B"), ("G1-e2b", "G1 graph rerank · E2B"),
        ("A-12b", "A Agentless-style · 12B"), ("A-e4b", "A Agentless-style · E4B"), ("A-e2b", "A Agentless-style · E2B"),
        ("B1-bm25", "B1 BM25"), ("RM-pagerank", "RM repo-map PageRank"), ("B2-embeddinggemma-2", "B2 EmbeddingGemma 2")]


def fig1() -> Path:
    d = json.loads((V01 / "results" / "final_eval.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    ys = list(range(len(ROWS)))[::-1]
    for y, (key, label) in zip(ys, ROWS):
        for metric, color, marker, dy in (("file.hit@1", BLUE, "o", 0.14), ("func.hit@10", ORANGE, "D", -0.14)):
            v = d[key]["all"][metric]
            ax.plot(v["ci"], [y + dy, y + dy], color=color, lw=2, solid_capstyle="round", alpha=0.55)
            ax.plot(v["mean"], y + dy, marker=marker, ms=6.5, color=color, mec=SURFACE, mew=1.5)
        if key.startswith("G1") or key == "B1-bm25":  # selective direct labels
            ax.text(1.005, y, f"{d[key]['all']['file.hit@1']['mean']:.2f} / {d[key]['all']['func.hit@10']['mean']:.2f}",
                    va="center", ha="left", color=INK2, fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_yticks(ys, [lab for _, lab in ROWS])
    ax.set_xlim(0.3, 0.95)
    ax.set_xlabel("Score on the evaluation set (n = 519, paired); bars = 95% repository-clustered bootstrap CI")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    h1 = ax.plot([], [], "o", color=BLUE, ms=6.5, label="File Hit@1")[0]
    h2 = ax.plot([], [], "D", color=ORANGE, ms=6.5, label="Function Hit@10")[0]
    ax.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "fig1_results.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2() -> tuple[Path, dict]:
    insts = {r["instance_id"]: r for r in load()}
    ids = (V01 / "eval_ids.txt").read_text(encoding="utf-8").split()[:400]
    pick = {"old": [i for i in ids if not insts[i]["born_after_cutoff"]
                    and not insts[i]["flags"]["issue_mentions_gold_file"]],
            "born": [i for i in ids if insts[i]["born_after_cutoff"] and not insts[i]["flags"]["issue_mentions_gold_file"]]}
    series = [("P-e2b", "Probe\nE2B"), ("P-e4b", "Probe\nE4B"), ("P-12b", "Probe\n12B"), ("R0-path", "Path-only\ncontrol")]
    vals = {}
    for key, _ in series:
        run = {r["instance_id"]: r for r in read_jsonl(V01 / "runs" / f"{key}.jsonl")}
        for grp, sub in pick.items():
            rows = {"x": [(insts[i]["repo"], hit_any(run[i]["files"], set(insts[i]["gold_files"]), 5)) for i in sub]}
            vals[(key, grp)] = cluster_boot(rows, lambda s: mean(s["x"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
    w = 0.36
    for k, (key, label) in enumerate(series):
        for grp, color, off, name in (("old", BLUE, -w / 2, "Old repositories"), ("born", ORANGE, w / 2, "Born-after repositories")):
            m, lo, hi = vals[(key, grp)]
            ax.bar(k + off, m, width=w - 0.04, color=color, label=name if k == 0 else None, zorder=2)
            ax.plot([k + off, k + off], [lo, hi], color=INK2, lw=1.2, zorder=3)
            ax.text(k + off, hi + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(series)), [lab for _, lab in series])
    ax.set_ylabel("File Hit@5")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title(f"Exploratory: T2 issues naming no gold file (old n={len(pick['old'])}, born-after n={len(pick['born'])})",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    path = OUT / "fig2_familiarity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, {f"{k}|{g}": [round(x, 4) for x in v] for (k, g), v in vals.items()}


if __name__ == "__main__":
    p1 = fig1()
    p2, v2 = fig2()
    (V01 / "results" / "fig2_values.json").write_text(json.dumps(v2, indent=1), encoding="utf-8")
    print(p1, p2)
    print(json.dumps(v2))
