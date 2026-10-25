# Pre-registration: Born After the Cutoff (dataset v0.1)

This file fixes the hypotheses, data splits and analysis plan **before** any Gemma 4 model is run on the evaluation set. It is committed to the local repository before those runs, and its commit hash goes in the paper.

Earlier Gemma calls were limited to:
- a 3-instance pipeline smoke test (24/10, `loc/smoke.py`, Gemma 4 E2B);
- the pilot described in §6, which is used only to fix prompts and parsing.

Any change made after the commit is logged in §8 with its date and reason.

## 1. Data

- **Instances:** `data/v0.1/gold.jsonl`, built by `harvest/harvest.py` and `harvest/gold.py` (selection rules in DESIGN.md §4).
- **Tiers** by issue creation date (UTC):
  - T0: before 2025-02-01;
  - T1: 2025-02-01 to 2026-04-01;
  - T2: from 2026-04-02 (Gemma 4 release).
  - For 12B we also report the strict tier: issues from 2026-06-04.
- **Born-after:** the repository was created on or after 2025-02-01.
- **Independent:** issue author ≠ fix author, issue at least 24 h before the fix, and the issue does not quote the added code.
- **Excluded from the evaluation strata, but kept in the release with flags:**
  - `mostly_new_code` instances, where more changed core lines sit in newly added files than in modified ones. Gold covers modified pre-existing code only, so the remaining gold files are often incidental. This rule came from a BM25 pilot on 40 a2a and openai-agents instances, before any Gemma run.
  - `issue_older_than_1y` instances, where the issue is more than 365 days older than its fix. The gold audit found a linking error among them.
  - Instances the gold audit judges wrong (AUDIT.md, `data/v0.1/audit_exclusions.txt`).
  - The OpenMinion slice is kept whole.
- **Evaluation set:** `harvest/select_eval.py --seed 0`, output `data/v0.1/eval_ids.txt` and `eval_manifest.json`. Strata:

  | Stratum | Content | Target size |
  |---|---|---|
  | `main` | T2; at most 60 per repository; half born-after; independent reports preferred | 400 |
  | `om` | OpenMinion case study (author-created; reported separately, never pooled) | all |
  | `rq2_t0` | T0, old repositories; at most 25 per repository | 100 |
  | `rq2_old` | T2, old repositories, topping up `main` to at least 100 | as needed |

  Final counts are recorded in §9 at commit time.

## 2. Methods compared

| Id | Method |
|---|---|
| R0-random | Random file order (seeded by instance id) |
| R0-path | BM25 over file paths only (guessability control) |
| B1 | BM25 over path and content (files), and over AST units (functions) |
| B2 | Dense retrieval: EmbeddingGemma 2 (256-d MRL), files ranked by their best unit. jina-embeddings-v2-base-code is secondary and runs only if compute allows; if it is skipped, that is reported |
| P | Probe: issue text only, no repository shown → 5 file paths |
| A | Agentless-style: core-file list → 5 files → skeletons of the top 3 → 10 functions |
| G1 | Graph rerank: B1 ∪ B2 seeds → 1-hop `calls`/`imports`/`contains` neighbours → listwise Gemma rerank |
| G2 | Graph agent: at most 12 tool steps; tools `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`, `read_file` |

**Models:** Gemma 4 E2B, E4B and 12B, instruction-tuned QAT q4_0 GGUF. 26B-A4B runs on a seeded 150-instance subsample of `main`. Temperature 0, seed 0, thinking off unless stated. A context budget of 16k tokens is primary.

## 3. Metrics

- **Primary:** file Hit@5 (at least one gold file in the top 5) and function Hit@10.
- **Secondary:**
  - file Acc@1 and Acc@5 (all gold in top-k);
  - MRR and NDCG@5;
  - function Acc@10;
  - prompt and completion tokens, wall-clock seconds and tool-call validity (G2).
- **Function name normalization:** done at prediction time against the snapshot's AST units. A bare name maps to `Class.name` only when it is unique in that file. Evaluation uses exact match.

## 4. Hypotheses and tests

**H1 (memory vs. skill, RQ2).**
- Probe P file Hit@5 is higher on `rq2_t0` (old repositories, pre-cutoff issues) than on born-after T2 instances.
- Define the gap as Δ = Hit@5(P) − Hit@5(R0-path) within each group. We predict Δ(T0-old) − Δ(born-after) > 0.
- Test: difference of within-group differences, with a repository-clustered bootstrap 95% CI (1,000 resamples, seed 0).
- **Support:** the CI excludes 0 for at least 2 of the 3 models.
- **Refutation:** the CI includes 0 for all models. That is reported as "no detectable memorization beyond guessability".

**H2 (graph helps small models on unfamiliar code, RQ3).** On `main`, for E2B and E4B: function Hit@10 is higher for G1 than for A, and the difference is larger on born-after instances than on old-repository instances.
- Test: a paired permutation test on per-instance differences (10,000 permutations); the interaction by bootstrap.
- Holm correction over the 2 models.

**H3 (scale, RQ3).** The advantage of G1 over A shrinks as size grows (E2B > E4B > 12B). This is descriptive: a monotonic trend in the point estimates, with CIs reported.

**H4 (Pareto, RQ4).** For E2B, B1 + G1 sits on the accuracy-vs-tokens Pareto front above G2: G1 has higher or equal function Hit@10 at lower median tokens. This is descriptive.

**H5 (lexical cues).** Identifier masking of the issue lowers A and B1 file Hit@5 more on instances whose issue mentions the gold file than on those that don't.
- Test: paired permutation test within each group, and the interaction by bootstrap.

**Main analyses** use the `independent ∩ main` subset when it has at least 150 instances, and `main` otherwise. Both are always reported. Results are given per repository (macro) and pooled (micro).

## 5. Exclusions

- Instances whose snapshot cannot be fetched or parsed are dropped and counted, never replaced.
- Model outputs that cannot be parsed count as empty rankings (score 0), never as missing.
- No instance is removed after looking at model results.
- Instances that the gold audit marks as wrong gold are excluded from all methods together. The list is fixed before the Gemma runs (AUDIT.md).

## 6. Pilot

At most 30 instances from T1 (not in the evaluation set) may be used to fix prompts, output parsing and tool-call formats. The pilot ids and the final prompt templates are committed with this file or logged in §8.

## 7. Stopping and budget

- Every model and method runs on the full strata listed above. Nothing stops early on results.
- If Kaggle GPU quota runs out, the remaining runs are reported as missing, and conclusions are restricted to the completed cells.

## 8. Deviations log

(empty at commit time)

## 9. Counts at commit time

- **Dataset v0.1** (`data/v0.1/gold.jsonl`): 4097 instances. By tier: {'T1': 1390, 'T2': 2425, 'T0': 282}.
  - Born-after: 1790. Independent: 1765 (of which T2: 790).
- **Evaluation set** (`eval_ids.txt`, seed 0):
  - `main`: n=400, born-after 200, independent 328, tiers {'T2': 400}
  - `om`: n=19, born-after 19, independent 0, tiers {'T2': 19}
  - `rq2_t0`: n=100, born-after 0, independent 57, tiers {'T0': 100}
  - `rq2_old`: n=0, born-after 0, independent 0, tiers {}
  - The `independent ∩ main` subset has 328 instances (≥ 150), so it is the primary analysis set (§4).
  - `rq2_old` is empty because `main` already holds 200 old-repository T2 instances (≥ 100); the RQ2 old-repository T2 cell is drawn from them.
- **Gold audit:** 57/60 gold-correct (95.0%, Wilson CI 86.3–98.3%). One exclusion (AUDIT.md).
- **Prompt templates:** frozen in `loc/run_llm.py` (P, A) and `loc/run_g1.py` (G1). G2 will be added before its runs and logged in §8.
- **Pilot ids:** none used yet. The T1 pilot (≤ 30 ids, none from the evaluation set) is logged in §8 when run.
