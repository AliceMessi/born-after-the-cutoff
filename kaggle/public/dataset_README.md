# Born After the Cutoff: agentloc v0.1

A code-localization benchmark for small Gemma 4 agents. Each instance pairs a GitHub issue with the
files and functions changed by the pull request or commit that fixed it. Instances are dated against
Gemma 4's training timeline, and part of them come from repositories created after its pre-training
cutoff.

- 4,097 issue-to-fix instances from 26 MIT, Apache-2.0 or BSD Python repositories, mostly agent and
  LLM frameworks.
- Tiers by issue date: T0 before 2025-02-01 (282), T1 up to Gemma 4's release on 2026-04-02 (1,390),
  T2 after it (2,425).
- 1,790 instances come from six repositories created after January 2025 (`born_after_cutoff`).
- An independent subset of 1,765 instances: reporter different from fixer, issue filed at least 24 h
  before the fix, no quoted fix code.
- A 60-instance audit, performed by an LLM-based research agent (Claude) without human review, found the
  gold labels correct for 57 (95.0%).

**Licence:** CC BY 4.0 for the dataset. The code copy in `code/` is Apache-2.0.

This dataset holds every run output and reference result behind the accompanying paper, so all of
its tables and figures can be rebuilt on CPU with the replay notebook.

## Files

| Path | Content |
|---|---|
| `instances.jsonl` | One record per instance: labels and metadata, no issue text (fields below) |
| `issues.jsonl` | Issue title and body, with @mentions, e-mail addresses, home-directory user names and credential-like strings (9 records) masked, plus `source_url`. The models saw the unmasked text |
| `stats.json` | Composition and harvest funnel per repository |
| `splits/` | Id lists: `eval_ids.txt` (519 pre-registered evaluation ids), `main_ids.txt` (400 T2 ids), `g2_ids.txt` (200-id agent subsample), `ids_26b.txt` (150 ids of the optional, not pre-registered 26B-A4B run, drawn from the G2 subsample), `pilot_ids.txt`, `pilot_g2_ids.txt`, `audit_ids.txt`, `audit_exclusions.txt`, `all_ids.txt`, `eval_manifest.json` (strata and seed) |
| `audit.jsonl` | The 60-instance gold audit: per-instance verdicts and a short note |
| `runs/` | Ranked predictions of every method on the evaluation set, with token counts and latency for the Gemma runs; the `*-26b` files cover `ids_26b.txt` only |
| `runs_full/` | R0 and BM25 rankings on all 4,097 instances |
| `runs_pilot/` | Outputs of the 30-instance pilot that fixed parsing and prompts before the evaluation runs |
| `results/` | Reference results: per-method metrics with bootstrap CIs, hypothesis tests, figure values, graph reachability |
| `env/` | Environment record of each Kaggle run: library versions, GPU, Ollama version, model digests |
| `graphs/` | Code graphs for the 519 evaluation instances at their base commit, without node text (see below) |
| `code/` | A copy of the scoring and analysis code (`loc/`, Apache-2.0) used by the replay notebook |
| `MANIFEST.sha256` | SHA-256 of every file as built, with the graphs as `graphs/<name>.json.gz` |
| `MANIFEST.graphs-json.sha256` | SHA-256 of each graph's decompressed JSON at `graphs/<name>.json/<name>.json`: Kaggle stores the uploaded `.json.gz` graphs decompressed under that path, and the replay notebook accepts either layout |

### Instance fields

- `instance_id`, `repo`, `source_kind` (`pr` or `commit`), `pr_number`, `issue_numbers`, `issue_url`, `fix_url`
- `issue_created_at`, `pr_created_at`, `merged_at`, `issue_to_fix_hours`
- `base_commit` (the repository state the task starts from), `merge_commit`, `diff_source`
- `tier` (`T0`, `T1`, `T2`), `after_12b_release` (issue filed on or after 2026-06-04, the 12B release),
  `born_after_cutoff`, `repo_created_at`
- `same_author` (issue author also authored the fix), `independent`
- `gold_files`: modified pre-existing core files of the landed diff
- `gold_functions`: `path::Qualified.name` of the innermost top-level function, class or method touched;
  a new method credits its class
- `added_symbols`, `module_level_change_files`, `core_lines_changed`, `test_files`
- `flags`: `issue_mentions_gold_file`, `issue_mentions_gold_function`, `issue_quotes_added_code`,
  `mostly_new_code`, `issue_older_than_1y`
- `filter_reason`

### Run files

One JSON object per line with `instance_id`, a ranked `files` list and, for most methods, a ranked
`functions` list. Gemma runs add `model`, `prompt_tokens`, `completion_tokens`, `seconds` and the raw
model reply; agent runs (G2) log each step's tool name and validity. Method names: R0-random,
R0-path, B1-bm25, B2-embeddinggemma-2, RM-pagerank, P (issue-only probe), A (Agentless-style),
G1 (graph rerank) and G2 (graph agent), each Gemma method suffixed with the model (e2b, e4b, 12b).
`*-masked` runs use issue text with code identifiers masked.

### Code graphs

NetworkX node-link JSON (`directed`, `multigraph`, `graph`, `nodes`, `edges`). Nodes carry `id`
(fully qualified dotted name), `name`, `type` (module, class, function or method), `file`,
`start_line` and `end_line`; edges carry `source`, `target`, `type` (`calls`, `contains`, `imports`, `inherits`) and `key`. Node text, which is
upstream source code, is omitted; the graph builder in the code repository restores it from the
repository at `base_commit`.

## Licences and attribution

- Labels, metadata, audit, run outputs, results and graphs (`instances.jsonl`, `splits/`, `audit.jsonl`,
  `runs*/`, `results/`, `env/`, `graphs/`, `stats.json`): CC BY 4.0.
- `issues.jsonl`: issue text was written by GitHub users and remains theirs. It is included for
  research convenience, masked, with a link to the source of every record. Open a discussion on the dataset page to
  have a record removed.
- `code/`: Apache-2.0.
- The source repositories keep their own licences (MIT, Apache-2.0 or BSD). No source code is
  redistributed; graph node names are identifiers taken from that code.
- Model outputs come from Gemma 4 (Apache-2.0) served by Ollama.
- No file from the Gemma 4 Developer Agent competition is included. Four repositories that the
  competition also uses were harvested independently from GitHub.

## Known limitations

Gold is the fix that landed, so other valid locations count as misses. An LLM-based research agent (Claude)
audited 60 instances, and no human reviewed its labels. Many issues name their own fix location; the `flags` and the independent subset let you
control for this. The 19 OpenMinion instances come from issues filed and fixed by the paper's author
with the same AI assistant, then reviewed and merged by the upstream maintainer; they are reported separately.
