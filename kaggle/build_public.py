"""Stage the public release locally (nothing is uploaded): release_public/{dataset,notebook,repo}.

  python -I harvest/build_release.py      (labels + masked issue text + stats)
  python -I kaggle/build_public.py        (assembles the staging folders)

dataset/   files for a future public Kaggle dataset: labels, masked issue text, id lists,
           audit sheet, every run output, reference results, environment records,
           text-free code graphs and a copy of loc/ for the replay notebook
notebook/  the CPU replay notebook
repo/      the code as it would be pushed to GitHub: tracked files minus internal notes,
           plus the public README, LICENSE, figures and new scripts (fresh git, one local commit)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V01 = ROOT / "data" / "v0.1"
OUT = ROOT / "release_public"
PUB = ROOT / "kaggle" / "public"          # hand-written public texts (README files, notebook source)
KOUT = ROOT / "data" / "kaggle_out"

ID_FILES = ["all_ids.txt", "eval_ids.txt", "main_ids.txt", "g2_ids.txt", "pilot_ids.txt", "pilot_g2_ids.txt",
            "audit_ids.txt", "audit_exclusions.txt", "eval_manifest.json"]
RESULTS = ["final_eval", "eval_e2b_e4b", "eval_12b", "g2_subsample", "masking", "eval_retrieval",
           "full_retrieval", "pilot_sanity", "hypotheses"]
REPO_EXCLUDE = {"README.md", "DESIGN.md"}  # internal planning notes stay private
REPO_EXTRA = ["loc/figures.py", "loc/verify_writeup.py", "kaggle/owner.py", "kaggle/build_public.py",
              "kaggle/graphs_slim/run.py", "figures/fig1_results.png", "figures/fig2_familiarity.png",
              "kaggle/public/dataset_README.md", "kaggle/public/repo_README.md", "kaggle/public/make_notebook.py",
              "kaggle/public/LICENSE"]


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_dataset() -> Path:
    ds = OUT / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    for name in ("instances.jsonl", "issues.jsonl", "stats.json"):
        copy(V01 / "release" / name, ds / name)
    for name in ID_FILES:
        copy(V01 / name, ds / "splits" / name)
    copy(V01 / "audit.jsonl", ds / "audit.jsonl")
    for sub in ("runs", "runs_full", "runs_pilot"):
        for f in sorted((V01 / sub).glob("*.jsonl")):
            copy(f, ds / sub / f.name)
    for stem in RESULTS:
        for ext in (".json", ".md"):
            if (V01 / "results" / (stem + ext)).exists():
                copy(V01 / "results" / (stem + ext), ds / "results" / (stem + ext))
    for name in ("graph_reachability_summary.json", "graph_reachability.jsonl", "fig2_values.json"):
        copy(V01 / "results" / name, ds / "results" / name)
    for env in sorted(KOUT.glob("*/out/env.json")):
        copy(env, ds / "env" / f"{env.parent.parent.name}.json")
    graphs = KOUT / "graphs-slim" / "out" / "graphs"
    if graphs.exists():
        (ds / "graphs").mkdir(parents=True, exist_ok=True)
        for g in sorted(graphs.glob("*.json.gz")):  # immutable downloads: hard links save disk space
            try:
                os.link(g, ds / "graphs" / g.name)
            except OSError:
                copy(g, ds / "graphs" / g.name)
    for p in sorted((ROOT / "loc").glob("*.py")):
        copy(p, ds / "code" / "loc" / p.name)
    copy(PUB / "dataset_README.md", ds / "README.md")
    copy(PUB / "LICENSE", ds / "code" / "LICENSE")
    lines = []
    for p in sorted(x for x in ds.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256"):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ds).as_posix()}")
    (ds / "MANIFEST.sha256").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    return ds


def build_notebook() -> Path:
    nb = OUT / "notebook"
    shutil.rmtree(nb, ignore_errors=True)
    nb.mkdir(parents=True)
    subprocess.run(["python", "-I", str(PUB / "make_notebook.py"), str(nb / "agentloc-replay.ipynb")], check=True)
    return nb


def build_repo() -> Path:
    repo = OUT / "repo"
    shutil.rmtree(repo, ignore_errors=True)
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.split()
    for rel in sorted(set(tracked) - REPO_EXCLUDE) + REPO_EXTRA:
        copy(ROOT / rel, repo / rel)
    copy(PUB / "repo_README.md", repo / "README.md")
    copy(PUB / "LICENSE", repo / "LICENSE")
    copy(OUT / "notebook" / "agentloc-replay.ipynb", repo / "notebooks" / "agentloc-replay.ipynb")
    git = ["git", "-C", str(repo), "-c", "user.name=agentloc", "-c", "user.email=agentloc@users.noreply.github.com"]
    subprocess.run(git[:3] + ["init", "-q", "-b", "main"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "Public release staging (local only)"], check=True)
    return repo


def size(p: Path) -> str:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".git" not in f.parts)
    n = sum(1 for f in p.rglob("*") if f.is_file() and ".git" not in f.parts)
    return f"{total / 1e6:.1f} MB in {n} files"


def main() -> None:
    ds = build_dataset()
    nb = build_notebook()
    repo = build_repo()
    for p in (ds, nb, repo):
        print(f"{p}: {size(p)}")
    print(json.dumps({d.name: size(d) for d in sorted(ds.iterdir()) if d.is_dir()}, indent=1))


if __name__ == "__main__":
    main()
