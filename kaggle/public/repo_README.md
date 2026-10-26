# Born After the Cutoff

A code-localization benchmark for small Gemma 4 agents, dated against the model's own training
timeline. This repository holds the harvester, gold extraction, graph builder, baselines, Gemma 4
runners and analysis behind the paper in `WRITEUP.md`.

Each instance pairs a GitHub issue with the files and functions changed by its fix. The 4,097
instances of v0.1 come from 26 permissively licensed Python repositories, mostly agent and LLM
frameworks. They are split into three tiers by issue date relative to Gemma 4's pre-training cutoff
(January 2025) and release (April 2026), and 1,790 of them come from six repositories created after
the cutoff, whose code the model cannot have seen.

## Layout

| Path | What it does |
|---|---|
| `harvest/` | GitHub harvest (PRs linked to the issues they close, a commit linker for Copybara-synced repositories, a curated mode), gold extraction from the landed diff, release builder, id selection |
| `graph/` | Code graph builder (NetworkX node-link JSON: `contains`, `imports`, `calls`, `inherits`), the three graph tools, reachability diagnostics |
| `loc/` | Baselines (R0, BM25, EmbeddingGemma 2, repo-map PageRank), Gemma 4 methods (P, A, G1, G2), scoring, hypothesis analysis, figures, the number check for the paper |
| `kaggle/` | Kaggle kernel scripts used for the dense and Gemma runs, the dataset builder, the public-release builder |
| `notebooks/agentloc-replay.ipynb` | CPU replay: rebuilds every table, test and figure from the released run outputs |
| `tests/` | 37 unit tests (`pytest`) |
| `PREREG.md` | Pre-registered hypotheses, strata, metrics and exclusions, with the deviation log (§8) |
| `DESIGN.md` | The working design note: research questions, data construction, methods, feasibility checks and step-by-step results |
| `AUDIT.md` | The 60-instance gold audit (performed by an LLM-based research agent, Claude, without human review): rubric, results, extractor bugs found and fixed |
| `WRITEUP.md`, `figures/` | The paper and its figures |

`DESIGN.md` is the working design note, kept with its full history so the order of design,
pre-registration and runs can be checked. Internal project-management passages (account names,
approvals, the publishing plan) were removed from every revision before publication; author dates
are unchanged.

## Reproduce

Python 3.11 or newer.

```bash
pip install -r requirements.txt
python -m pytest -q
```

**From the released outputs (CPU, a few minutes).** Download the dataset, then run
`notebooks/agentloc-replay.ipynb` with `AGENTLOC_PUBLIC` pointing at it. The notebook checks every file
against `MANIFEST.sha256`, re-scores every run file with the code in `loc/`, re-runs the hypothesis
analysis and figure script, and reports any difference from the reference results.

**From scratch.** The harvester uses an existing GitHub CLI login (`gh`) and caches every response
under `data/cache/` so a harvest can be re-run offline.

```bash
python -I harvest/run_all.py harvest/repos_v01.json --out data/v0.1/raw
python -I harvest/gold.py data/v0.1/raw/*.candidates.jsonl --out data/v0.1/gold.jsonl
python -I harvest/select_eval.py --seed 0     # the pre-registered evaluation ids
python -I harvest/build_release.py            # labels, masked issue text, stats
python -I loc/run_baselines.py --ids-file data/v0.1/eval_ids.txt
python -I loc/run_rm.py --ids-file data/v0.1/eval_ids.txt
python -I loc/run_dense.py --model embeddinggemma-2 --backend st --served-id google/embeddinggemma-2 --ids-file data/v0.1/eval_ids.txt
python -I loc/evaluate.py --name final_eval
python -I loc/analysis.py
python -I loc/figures.py
```

Repository snapshots are cloned into `data/repos/` (override with `AGENTLOC_REPO_STORE`). Everything
under `data/` stays local.

**Gemma 4 runs.** `loc/run_llm.py` (P, A), `loc/run_g1.py` (G1) and `loc/run_g2.py` (G2) talk to an
OpenAI-compatible server or to Ollama (`--backend ollama`). The paper used Ollama 0.40.11 on one Kaggle
T4 with `gemma4:e2b-it-qat`, `gemma4:e4b-it-qat` and `gemma4:12b-it-qat`, a 16k context and thinking
off. `kaggle/gemma_eval/run.py` is the kernel that ran them; `kaggle/push.py` builds a kernel folder
(set `KAGGLE_OWNER` to your Kaggle username). Model digests and library versions of every run are in
the dataset's `env/` folder.

## Data and licences

- Code in this repository: Apache-2.0 (`LICENSE`).
- Dataset (labels, run outputs, results, text-free code graphs): CC BY 4.0. Issue text is included
  masked, for research use, with a source link per record; it remains its authors'.
- Code graphs ship without node text. Rebuild it with `graph/build_graph.py` from the repository at the
  instance's `base_commit`.
- No file from the Gemma 4 Developer Agent competition is included or read by this code. Four
  repositories the competition also uses were harvested independently from GitHub.

## AI assistance

The code, the analyses, the gold audit and the drafting of the paper were done with an AI coding and
research assistant (Claude), directed and reviewed by the authors.

## Citation

If you use the benchmark, please cite the paper in `WRITEUP.md` and link this repository.
