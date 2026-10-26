# Born After the Cutoff

### A contamination-controlled localization benchmark for small Gemma 4 coding agents, dated against the model's own training timeline

## Abstract

Small local models must find *where* to change code before they can fix anything, but localization benchmarks are built from repositories that today's models have likely seen during training. We release a resource that controls this on two levels.

- **Time.** 4,097 issue→fix instances from 26 permissively licensed Python agent and LLM-framework repositories are split into three tiers by issue date, relative to Gemma 4's stated pre-training cutoff (January 2025) and public release (April 2026).
- **Codebase.** 1,790 instances come from six repositories that did not exist before the cutoff.

**Gold quality.** Gold files and functions come from the landed diff. An audit of 60 instances finds 95% of gold labels correct (Wilson 95% CI 86–98%).

**Format.** We ship the tasks with graphs in the same NetworkX node-link layout the Gemma 4 Developer Agent competition documents publicly. We also provide no-LLM baselines (BM25, dense retrieval with EmbeddingGemma 2, and a repo-map PageRank) and four Gemma 4 methods.

**Results.** On a pre-registered 519-instance evaluation set, a single-call graph rerank (G1) with Gemma 4 E4B reaches file Hit@1 0.80 and function Hit@10 0.80, against 0.56 and 0.71 for BM25; 12B reaches 0.83 and 0.81.
- G1 beats an Agentless-style prompt by +0.23 function Hit@10 for both E2B and E4B (Holm-corrected p = 0.0002).
- G1 beats a 12-step tool-using graph agent at a tenth of the tokens.

**Two pre-registered predictions failed, and we report them as such.** The graph's gain is not larger on unseen codebases, and a memorization probe shows no tier effect beyond guessability. An exploratory analysis, labelled as such, points to a codebase-familiarity effect that only the born-after split can expose.

## 1. Problem and motivation

Localization (choosing files and functions to edit) is the first step of every issue-fixing agent, and errors there cascade. Agentless [1], LocAgent [2], CoSIL [3] and SweRank [4] show how strongly the final result depends on it.

**The contamination problem.** For small local models it is hard to measure cleanly. Public benchmarks reuse popular repositories, and models can recall file paths from issue text alone [5] or lean on familiar repository structure [6]. Live collections such as SWE-bench-Live [7] and SWE-rebench [8] refresh tasks over time, but they score resolution, not localization. They are also dated by calendar, not by a specific model's training timeline.

**The competition setting.** The Gemma 4 Developer Agent competition provides code graphs and a similarity tool over pre-computed embeddings. The public page states that the tool "finds top-k graph nodes with highest cosine similarity to `query` in the pre-computed embeddings" and powers "offline semantic code retrieval". Computed on the main track's public dev data, most of the 129 dev tasks were created after Gemma 4's pre-training cutoff, while all four dev repositories were created between 2011 and 2019.

Agents tuned on those public tasks therefore work in familiar codebases, while the hidden test set comes from private repositories. A public set of *unfamiliar* codebases lets participants and researchers measure the gap.

**Our questions:**
- Can such a set be built automatically and audited?
- Does graph structure help small Gemma 4 models localize?
- Can we separate memory from skill?

## 2. Related work

- **Resolution benchmarks.** SWE-bench [9] and its live successors [7, 8] score resolution. SWE-Bench+ [10] reports solution leakage in issue text.
- **Localization benchmarks.** Loc-Bench [2] (560 instances, bugs after October 2024), Long Code Arena's bug localization [11] and SWE-PolyBench [12] report retrieval metrics.
- **Small models.** IssueLoc-Bench [13] and SemNav [14] already run Gemma 4 E4B as a localizer.
- **Agent-framework repositories.** AgentIssue-Bench [15] studies agent-framework issues but evaluates resolution only.
- **Methods.** Graph-guided localization includes LocAgent [2], CoSIL [3] and RepoGraph [16]. Our RM baseline is inspired by aider's repo-map [17].
- **Our contribution** is the combination: model-dated tiers, a codebase-level control, the agent-framework domain, a format compatible with the competition, and a pre-registered analysis.

## 3. The resource

**Sources.** 26 MIT, Apache-2.0 or BSD repositories:
- agent and LLM frameworks: openai-agents-python, pydantic-ai, fastmcp, microsoft/agent-framework, langgraph, crewAI, dspy, haystack, livekit agents, pipecat and others;
- the competition's four public repositories, harvested independently from GitHub.

We never redistribute any competition file.

**Born-after repositories.** Six repositories were created after January 2025: openai-agents-python, strands, agent-framework, a2a-python, adk-python and OpenMinion.

**Harvesting.** Merged PRs are linked to the issue they close. GitHub links these through `closingIssuesReferences`, and we fall back to closing keywords in the PR body. Copybara-synced repositories close issues by direct commits, so a commit linker adds adk-python (498 instances).

**Filters:**
- the issue predates the fix, and neither issue nor PR is authored by a bot;
- 1–5 modified core files and at most 400 changed core lines;
- tests are changed;
- no trivial or documentation-only fixes;
- the issue does not contain the diff.

**Gold.**
- *Files:* the modified pre-existing core files of the landed diff, i.e. the merge commit against its first parent.
- *Functions:* changed old-side lines map, with Python's `ast`, to the innermost top-level function, class or method.
- *New code:* new methods credit their class, and nested helpers credit their enclosing function.
- *Flags:* instances where most changed code is new are flagged (2.7%), as are issues more than a year older than their fix.

**Tiers** by issue date:
- **T0**, before 2025-02-01: 282 instances;
- **T1**, up to the release: 1,390;
- **T2**, after 2026-04-02: 2,425.

Post-training data cannot contain T2 issues.

**Provenance flags.** The reporter is also the fixer in 36% of instances, and many issues are filed minutes before the fix. Issues name a gold file in 41% of instances and a gold function in 60%. The **independent** subset (different author, filed at least 24 h before the fix, no quoted fix code) has 1,765 instances.

**Audit.** One annotator labelled 60 stratified instances against a fixed rubric.
- Gold was correct for 57/60 instances (95.0%, CI 86.3–98.3%). There was one linking error and one new-code case.
- 55% of the issues are *prescriptive*: they name the exact location or change.
- The audit also found four extractor bugs: an appended function credited to its neighbour, stacked-PR contamination, re-wrapped signatures, and definitions inside module-level `try` blocks. Each was fixed with a regression test, and all gold was regenerated before any model run.

**Format.** We release:
- tasks (labels without issue text, under CC BY 4.0, with source URLs);
- issue text with mentions and e-mail addresses masked;
- code graphs in NetworkX node-link JSON, with the publicly documented fields `id`/`name`/`text` and `source`/`target`/`type`/`key`, plus module nodes and typed edges (calls, contains, imports, inherits);
- the harvester, so new tiers can be built for future models.

## 4. Tasks, metrics and methods

**Task.** Given an issue and the repository at the base commit, rank files and functions.

**Metrics.**
- Hit@k: at least one gold item in the top k.
- Acc@k: all gold items in the top k.
- MRR.
- Cost: prompt tokens and seconds.

CIs come from 1,000 repository-clustered bootstrap resamples.

**No-LLM baselines:**
- **R0:** random order, and BM25 over file paths only (a guessability control).
- **B1:** BM25 over file content and over AST units.
- **B2:** EmbeddingGemma 2 [18] at 256 dimensions (MRL truncation), with files ranked by their best unit.
- **RM:** personalized PageRank over our code graph, seeded by code-like identifiers in the issue.

**Gemma 4 methods** [19]. E2B, E4B and 12B, run as instruction-tuned QAT checkpoints through Ollama on one Kaggle T4, with a 16k context and thinking off:
- **P:** issue only, no repository shown → 5 paths. This is a memorization probe in the spirit of [5].
- **A:** Agentless-style [1]: file list → 5 files → skeletons of the top 3 → functions.
- **G1:** graph rerank. B1 and B2 seeds are expanded by one hop over calls, inherits and contains edges into a pool of at most 40 units, which Gemma reranks in one call.
- **G2:** a graph agent with the competition's three graph-tool signatures plus `read_file`, at most 12 steps.

**Graph headroom.** From the B1 top-10 seeds, one hop reaches 58.2% of gold functions, against 38.6% at zero hops, with a median pool of 59 units.

**Pre-registration.** Hypotheses, strata, metrics and exclusions were committed before any Gemma run on evaluation instances. Changes made after the 30-instance pilot (output parsing, prompt examples, G2 requiring at least two tool calls) and a 200-instance G2 subsample for budget reasons were logged before the evaluation runs.

The evaluation set has 519 instances:
- `main`: 400 T2 instances (200 born-after; 328 independent);
- 19 hand-verified OpenMinion instances, from issues the authors filed and fixed, reported separately;
- 100 T0 instances from older repositories.

## 5. Results

**No-LLM baselines** (n = 519, paired)

| Method | File Hit@1 | File Hit@5 | Function Hit@10 |
|---|---|---|---|
| Random | 0.019 | 0.054 | – |
| Path-only BM25 | 0.266 | 0.586 | – |
| RM (graph, no LLM) | 0.484 | 0.800 | 0.635 |
| B2 EmbeddingGemma 2 | 0.437 | 0.744 | 0.517 |
| B1 BM25 | 0.557 | 0.800 | 0.714 |

- BM25 is a strong floor. RM matches it at Hit@5 using only the graph and identifiers from the issue.
- Dense retrieval loses overall but wins on the older T0 issues (Hit@5 0.730 vs 0.650), where lexical cues are scarcer.
- On the full 4,097 instances, BM25 Hit@5 drops from T2 (0.854) to T1 (0.725) and T0 (0.610).

**Gemma 4** (n = 519, paired)

| Method | File Hit@1 | File Hit@5 | Function Hit@10 | Median prompt tokens |
|---|---|---|---|---|
| P, E2B | 0.516 | 0.572 | – | 864 |
| A, E2B | 0.613 | 0.805 | 0.563 | 8,096 |
| **G1, E2B** | **0.748** | **0.871** | **0.760** | 3,063 |
| P, E4B | 0.532 | 0.570 | – | 864 |
| A, E4B | 0.653 | 0.813 | 0.614 | 7,938 |
| **G1, E4B** | **0.801** | **0.896** | **0.797** | 3,063 |
| P, 12B | 0.561 | 0.609 | – | 868 |
| A, 12B | 0.740 | 0.877 | 0.706 | 8,061 |
| **G1, 12B** | **0.827** | **0.911** | **0.811** | 3,067 |

On the 200-instance G2 subsample:

| Method | File Hit@1 | File Hit@5 | Function Hit@10 | Median prompt tokens |
|---|---|---|---|---|
| B1 BM25 | 0.585 | 0.825 | 0.755 | 0 |
| G1, E2B / E4B / 12B | 0.810 / 0.840 / 0.845 | 0.900 / 0.920 / 0.940 | 0.805 / 0.825 / 0.840 | about 3.1k |
| G2, E2B / E4B | 0.580 / 0.750 | 0.600 / 0.780 | 0.365 / 0.465 | 26.6k / 33.9k |
| G2, 12B | *[pending]* | | | |

- G1 is the strongest method in every subset. That includes issues that name no gold file, where G1 with E4B scores file Hit@1 0.717 against 0.450 for BM25.
- G1 needs fewer than half of A's tokens: about 1.6 s per instance for E2B, 2.8 s for E4B and 9.2 s for 12B on a T4.
- **The G2 agent is far weaker and costlier than the single-call rerank, even though 86% (E2B) and 97% (E4B) of its tool calls are valid.** Small models stop exploring early and name few functions.

**Pre-registered hypotheses**

- **H1, memory beyond guessability: not supported.** We tested whether the probe's margin over the path-only control is larger for old-repository T0 issues than for born-after T2 issues.
  - The difference is −0.115 for E2B, −0.125 for E4B and −0.060 for 12B, and every CI includes 0.
  - The probe's success tracks file names written in the issue, which T2 issues contain more often.
- **H2, graph helps: supported.** On independent ∩ main (n = 325), G1 beats A on function Hit@10: 0.785 vs 0.551 for E2B and 0.840 vs 0.612 for E4B, with Holm p = 0.0002.
  - **The predicted larger gain on born-after repositories was not observed:** the interaction is −0.048, with CIs spanning 0.
- **H3, the gain shrinks with size (descriptive): observed.** G1 − A is 0.210 for E2B, 0.203 for E4B and 0.129 for 12B (CI 0.077–0.190). 12B's Agentless prompt closes part of the gap.
- **H4, Pareto: supported.** On the G2 subsample, G1 with E2B reaches function Hit@10 0.805 at 3.1k median tokens. That beats G2 (0.365 at 26.6k) and A (0.580 at 8.6k).
- **H5, lexical cues:** identifier masking cuts BM25's file Hit@5 by 0.39–0.41 whether or not the issue names the file (no interaction). *[A with E4B pending]*

**Exploratory, not pre-registered.** We restricted the probe to T2 issues that name no gold file, so issue dates are comparable and the obvious lexical cue is removed.

| Repositories | n | Probe file Hit@5, E2B / E4B / 12B | Path-only control Hit@5 |
|---|---|---|---|
| Old | 86 | 0.384 / 0.360 / 0.419 | 0.605 |
| Born-after | 91 | 0.176 / 0.198 / 0.242 | 0.538 |

Without seeing the repository, all three models guess files markedly better in long-established codebases (+0.16 to +0.21), while the path-only control differs by +0.07. This is the familiarity confound that a date-only split cannot remove and that the born-after control is designed to measure.

## 6. Limitations

- **Audit.** It was done by a single annotator who also wrote the pipeline. Every error found was fixed in code, not relabelled.
- **Gold is the landed fix.** Valid alternative locations count as misses. Hit@k softens this; Acc@k does not.
- **Sampling.** The G2 subsample (200) and the T0 cell (100) limit power. T0 also mixes older, terser issues, which confounds tier with issue style.
- **Prescriptive issues.** Many issues name their own fix location. We report the independent and no-mention subsets for this reason.
- **Quantization.** Results use 4-bit QAT GGUF checkpoints served by Ollama. The competition harness serves different quantized checkpoints, so absolute numbers may shift.
- **OpenMinion.** The 19 hand-verified instances come from issues the authors filed and fixed. They are analysed separately, never pooled.

## 7. Verifiability

All code is released under Apache-2.0: harvester, gold extraction, graph builder, baselines, Gemma runners and analysis. There are 35 unit tests.

**What we record:**
- the dataset with labels, issue text, flags, the audit sheet and the evaluation, pilot and G2 id lists;
- every model output with token counts and latency;
- exact model tags and digests, and pinned library versions;
- seeds and prompt templates.

A public Kaggle notebook rebuilds every table from the released outputs on CPU, and re-runs a 50-instance Gemma subset on a T4. The pre-registration and its deviation log are versioned with commit hashes.

## References

1. Xia et al. *Agentless: Demystifying LLM-based Software Engineering Agents.* FSE 2025. https://arxiv.org/abs/2407.01489
2. Chen et al. *LocAgent: Graph-Guided LLM Agents for Code Localization.* ACL 2025. https://arxiv.org/abs/2503.09089
3. Jiang et al. *CoSIL: Issue Localization via LLM-Driven Iterative Code Graph Searching.* ASE 2025. https://arxiv.org/abs/2503.22424
4. Reddy et al. *SweRank: Software Issue Localization with Code Ranking.* ICLR 2026. https://arxiv.org/abs/2505.07849
5. Liang et al. *The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason.* 2025. https://arxiv.org/abs/2506.12286
6. Chen et al. *Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?* 2026. https://arxiv.org/abs/2609.27891
7. Zhang et al. *SWE-bench Goes Live!* 2025. https://arxiv.org/abs/2505.23419
8. Badertdinov et al. *SWE-rebench.* NeurIPS 2025 Datasets & Benchmarks. https://arxiv.org/abs/2505.20411
9. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. https://arxiv.org/abs/2310.06770
10. Aleithan et al. *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* 2024. https://arxiv.org/abs/2410.06992
11. Bogomolov et al. *Long Code Arena.* 2024. https://arxiv.org/abs/2406.11612
12. Rashid et al. *SWE-PolyBench.* 2025. https://arxiv.org/abs/2504.08703
13. Al Awad & Ivanov. *Cost-Effective Repository Exploration for Agentic Issue Localization.* 2026. https://arxiv.org/abs/2608.29675
14. Wei et al. *Semantic Navigation for Issue Localization in Code Repository.* 2026. https://arxiv.org/abs/2609.31176
15. Rahardja et al. *Can Agents Fix Agent Issues?* NeurIPS 2025. https://arxiv.org/abs/2505.20749
16. Ouyang et al. *RepoGraph.* ICLR 2025. https://arxiv.org/abs/2410.14684
17. Aider. *Repository map.* https://aider.chat/docs/repomap.html
18. Google DeepMind. *EmbeddingGemma 2* model card, 2026. https://huggingface.co/google/embeddinggemma-2
19. Gemma Team. *Gemma 4 Technical Report.* 2026. https://arxiv.org/abs/2607.02770
