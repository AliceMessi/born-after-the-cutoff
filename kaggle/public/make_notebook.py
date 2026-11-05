"""Write the CPU replay notebook (plain nbformat-4 JSON, no dependencies).

  python -I kaggle/public/make_notebook.py release_public/notebook/agentloc-replay.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS = [
    ("markdown", """# Born After the Cutoff: CPU replay

This notebook rebuilds the paper's tables, hypothesis tests and figures from the released run outputs.
It needs no GPU and no internet: every number comes from the files in the dataset, scored by the same
code (`code/loc/`) that produced the paper.

Steps:
1. find the dataset and check every file against `MANIFEST.sha256`;
2. lay the files out the way the scoring code expects;
3. re-score every run file (repository-clustered bootstrap, 1,000 resamples, seed 0);
4. re-run the pre-registered hypothesis analysis and the figure script;
5. compare everything with the reference results shipped in `results/`.

The Gemma runs themselves (Ollama on a Kaggle T4) are not repeated here; their run scripts are in
the code repository."""),
    ("code", """import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

def find_dataset() -> Path:
    if os.environ.get("AGENTLOC_PUBLIC"):
        return Path(os.environ["AGENTLOC_PUBLIC"]).resolve()
    for root in (Path("/kaggle/input"), Path.cwd().parent / "dataset", Path.cwd()):
        hits = sorted(root.glob("**/MANIFEST.sha256")) if root.exists() else []
        if hits:
            return hits[0].parent
    raise FileNotFoundError("dataset not found: set AGENTLOC_PUBLIC to the folder with MANIFEST.sha256")

DATA = find_dataset()
WORK = Path("/kaggle/working/agentloc") if Path("/kaggle/working").exists() else Path.cwd() / "agentloc_work"
print("dataset:", DATA)
print("work dir:", WORK)"""),
    ("code", """bad = []
lines = (DATA / "MANIFEST.sha256").read_text(encoding="utf-8").split(chr(10))
for line in filter(None, lines):
    digest, rel = line.split("  ", 1)
    if hashlib.sha256((DATA / rel).read_bytes()).hexdigest() != digest:
        bad.append(rel)
print(f"{len(lines) - 1} files checked, {len(bad)} mismatches", bad[:5])"""),
    ("code", """# The scoring code expects data/v0.1/{release,runs,...} next to loc/.
shutil.rmtree(WORK, ignore_errors=True)
V01 = WORK / "data" / "v0.1"
shutil.copytree(DATA / "code" / "loc", WORK / "loc")
for name in ("instances.jsonl", "issues.jsonl", "stats.json"):
    (V01 / "release").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / name, V01 / "release" / name)
for f in (DATA / "splits").iterdir():
    shutil.copy(f, V01 / f.name)
for sub in ("runs", "runs_full", "runs_pilot"):
    shutil.copytree(DATA / sub, V01 / sub)
(V01 / "results").mkdir()
REF = DATA / "results"
print(sorted(p.name for p in V01.iterdir()))"""),
    ("code", """# Re-score each reference result with exactly its methods and instance ids.
JOBS = {
    "final_eval": ("runs", "eval_ids.txt"),
    "eval_retrieval": ("runs", "eval_ids.txt"),
    "g2_subsample": ("runs", "g2_ids.txt"),
    "eval_26b": ("runs", "ids_26b.txt"),  # optional 26B-A4B run, not pre-registered
    "masking": ("runs", "main_ids.txt"),
    "full_retrieval": ("runs_full", "all_ids.txt"),
}
def run(*args):
    p = subprocess.run([sys.executable, *args], cwd=WORK, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise RuntimeError(args)
    return p.stdout

for name, (runs, ids) in JOBS.items():
    methods = ",".join(json.loads((REF / f"{name}.json").read_text(encoding="utf-8")))
    run("loc/evaluate.py", "--runs", str(V01 / runs), "--ids-file", str(V01 / ids),
        "--name", name, "--methods", methods)
    print("scored", name)
run("loc/analysis.py")
run("loc/figures.py")
print("hypothesis analysis and figures rebuilt")"""),
    ("code", """def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:  # both NaN
        return []
    return [] if a == b else [f"{path}: {a} != {b}"]

for name in list(JOBS) + ["hypotheses", "fig2_values"]:
    new = json.loads((V01 / "results" / f"{name}.json").read_text(encoding="utf-8"))
    ref = json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))
    d = diff(new, ref)
    print(f"{name:16s} {'identical' if not d else str(len(d)) + ' differences'}", d[:3])"""),
    ("code", """# Table 1: paired results on the 519-instance evaluation set.
ev = json.loads((V01 / "results" / "final_eval.json").read_text(encoding="utf-8"))
rows = ["R0-random", "R0-path", "B1-bm25", "B2-embeddinggemma-2", "RM-pagerank",
        "P-e2b", "P-e4b", "P-12b", "A-e2b", "A-e4b", "A-12b", "G1-e2b", "G1-e4b", "G1-12b"]
cols = [("file.hit@1", "file Hit@1"), ("file.hit@5", "file Hit@5"), ("func.hit@10", "function Hit@10")]
print(f"{'method':22s}" + "".join(f"{c:>26s}" for _, c in cols))
for m in rows:
    cells = []
    for k, _ in cols:
        v = ev[m]["all"].get(k)
        cells.append(f"{v['mean']:.3f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]" if v else "-")
    print(f"{m:22s}" + "".join(f"{c:>26s}" for c in cells))"""),
    ("code", """# Pre-registered hypotheses (PREREG.md); full detail in results/hypotheses.json.
hy = json.loads((V01 / "results" / "hypotheses.json").read_text(encoding="utf-8"))
ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
print("H1 probe margin over path control, T0-old minus born-after T2:", hy["H1"]["verdict"])
for m, v in hy["H1"]["per_model"].items():
    print(f"   {m}: {v['diff_of_diffs']:+.3f} {ci(v['ci'])}")
print("H2 G1 vs A, function Hit@10 on", hy["H2"]["primary_subset"])
for m, v in hy["H2"]["per_model"].items():
    print(f"   {m}: n={v['n']} G1 {v['G1']:.3f} A {v['A']:.3f} Holm p {v['p_holm']:.4f};"
          f" born-after interaction {v['interaction_born_minus_old']:+.3f} {ci(v['interaction_ci'])}")
print("H3 G1 - A by model size, monotonic decrease:", hy["H3"]["monotonic_decrease"])
for m, v in hy["H3"]["per_model"].items():
    print(f"   {m}: {v['G1_minus_A']:+.3f} {ci(v['ci'])}")
print("H4 accuracy vs tokens on the G2 subsample, G1 dominates G2:", hy["H4"]["G1_dominates_G2"])
for m, v in hy["H4"]["points"].items():
    print(f"   {m}: function Hit@10 {v['func_hit10']:.3f} at {v['median_prompt_tokens']} median prompt tokens")
print("H5 identifier masking, file Hit@5 drop (issue names the gold file / does not):")
for m, v in hy["H5"].items():
    print(f"   {m}: {v['drop_mention']:.3f} / {v['drop_no_mention']:.3f}; interaction {v['interaction']:+.3f} {ci(v['interaction_ci'])}")"""),
    ("code", """from IPython.display import Image, display
for f in ("fig1_results.png", "fig2_familiarity.png"):
    display(Image(filename=str(WORK / "figures" / f)))"""),
    ("code", """# Code graphs: NetworkX node-link JSON without node text (upstream source code is not redistributed).
import gzip
graphs = sorted((DATA / "graphs").glob("*.json.gz"))
with_text = 0
for g in graphs:
    with gzip.open(g, "rt", encoding="utf-8") as fh:
        G = json.load(fh)
    with_text += any("text" in n for n in G["nodes"])
print(len(graphs), "graphs;", with_text, "contain node text")
if graphs:
    with gzip.open(graphs[0], "rt", encoding="utf-8") as fh:
        G = json.load(fh)
    print(graphs[0].name, {k: G[k] for k in ("directed", "multigraph")}, len(G["nodes"]), "nodes", len(G["edges"]), "edges")
    print("node fields:", sorted(G["nodes"][0]), "edge fields:", sorted(G["edges"][0]))"""),
    ("markdown", """To restore node text, clone the repository at the instance's `base_commit` and run
`graph/build_graph.py` from the code repository; node ids and edges are deterministic for a given commit."""),
]


def main() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(sys.argv[1])


if __name__ == "__main__":
    main()
