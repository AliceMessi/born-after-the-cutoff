"""Stage the public Kaggle dataset and notebook locally (nothing is uploaded).

  python -I harvest/build_release.py      (labels + masked issue text + stats)
  python -I kaggle/build_public.py        (assembles release_public/dataset and release_public/notebook)

dataset/   files for the public Kaggle dataset: labels, masked issue text, id lists, audit sheet,
           every run output, reference results, environment records, text-free code graphs and a
           copy of loc/ for the replay notebook, plus dataset-metadata.json (CC BY 4.0)
notebook/  the CPU replay notebook and its kernel-metadata.json (staged as private until publishing day)

The GitHub repository is built separately from this project's filtered history.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
from owner import owner  # noqa: E402

V01 = ROOT / "data" / "v0.1"
OUT = ROOT / "release_public"
PUB = ROOT / "kaggle" / "public"          # hand-written public texts (README files, notebook source)
KOUT = ROOT / "data" / "kaggle_out"
DATASET_SLUG = "born-after-the-cutoff"
NOTEBOOK_SLUG = "born-after-the-cutoff-replay"

ID_FILES = ["all_ids.txt", "eval_ids.txt", "main_ids.txt", "g2_ids.txt", "pilot_ids.txt", "pilot_g2_ids.txt",
            "audit_ids.txt", "audit_exclusions.txt", "eval_manifest.json"]
RESULTS = ["final_eval", "eval_e2b_e4b", "eval_12b", "g2_subsample", "masking", "eval_retrieval",
           "full_retrieval", "pilot_sanity", "hypotheses"]
CODE_NOTE = """This folder is a copy of `loc/` from the code repository, included so the replay notebook runs
offline. Licence: Apache-2.0 (see LICENSE). The rest of the dataset is CC BY 4.0.
"""


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
    copy(PUB / "LICENSE", ds / "code" / "LICENSE")
    (ds / "code" / "README.md").write_text(CODE_NOTE, encoding="utf-8")
    copy(PUB / "dataset_README.md", ds / "README.md")
    lines = []
    for p in sorted(x for x in ds.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256"):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ds).as_posix()}")
    (ds / "MANIFEST.sha256").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    # Kaggle reads this file as metadata; it is written after the manifest so the manifest
    # lists only data files.
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": "Born After the Cutoff: agentloc v0.1",
        "id": f"{owner()}/{DATASET_SLUG}",
        "subtitle": "Contamination-controlled code-localization benchmark for small Gemma 4 agents",
        "description": (PUB / "dataset_README.md").read_text(encoding="utf-8"),
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["code", "benchmark", "nlp", "computer science"],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    return ds


def build_notebook() -> Path:
    nb = OUT / "notebook"
    shutil.rmtree(nb, ignore_errors=True)
    nb.mkdir(parents=True)
    subprocess.run([sys.executable, "-I", str(PUB / "make_notebook.py"), str(nb / "agentloc-replay.ipynb")], check=True)
    (nb / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner()}/{NOTEBOOK_SLUG}", "title": "Born After the Cutoff: CPU replay",
        "code_file": "agentloc-replay.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True,  # flipped to false only on publishing day
        "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [f"{owner()}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": [],
        "model_sources": [],
    }, indent=1), encoding="utf-8")
    return nb


def size(p: Path) -> str:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    n = sum(1 for f in p.rglob("*") if f.is_file())
    return f"{total / 1e6:.1f} MB in {n} files"


def main() -> None:
    ds = build_dataset()
    nb = build_notebook()
    for p in (ds, nb):
        print(f"{p}: {size(p)}")
    print(json.dumps({d.name: size(d) for d in sorted(ds.iterdir()) if d.is_dir()}, indent=1))


if __name__ == "__main__":
    main()
