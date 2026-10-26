# Research design: Gemma 4 Developer Agent Paper Track

Status: design + feasibility, 24 Oct 2026. Nothing has been published. Scripts live in `harvest/` and `loc/`. Dry-run data lives in `data/` (local only, not for release as is).


**Working title:** *Born After the Cutoff: a contamination-controlled, graph-native localization benchmark for small Gemma 4 coding agents.*
- The dataset name is still to be chosen (working name **AgentLoc-Fresh**). Check for name collisions before release.

---

## 0. Recommendation in one paragraph

Keep the localization-resource idea, but change the angle. The original pitch was "a post-cutoff localization set evaluated with small Gemma models". That is no longer new on its own: IssueLoc-Bench [27] already runs Gemma-4-E4B locally on recent issue→PR pairs with Hit@k, MRR, tokens and time, and SemNav [28] uses Gemma 4 E4B as a localizer.

The defensible, judge-relevant contribution has four parts:

1. **A three-tier, model-dated split keyed to Gemma 4's stated dates.**
   - Gemma 4's pre-training cutoff is January 2025; the public release was 2 April 2026.
   - The split adds **repositories that were born after the cutoff**. Their code cannot have been seen in pre-training, which separates "knowing the repository" from "localizing in it".
2. **Agent/LLM-framework repositories as the domain.**
   - No localization benchmark covers them. The closest work is AgentIssue-Bench [33]: 50 tasks, resolution only.
3. **A drop-in format for the main competition.**
   - Tasks, NetworkX AST call graphs and 256-d node embeddings follow the competition's published schema.
   - Graph tools use exactly the competition's signatures (`get_code_neighbors`, `search_similar_code`, `get_code_subgraph`). Any main-track agent can run on our data unchanged.
4. **Three measured findings:**
   - (a) a memorization probe with a guessability control, run across tiers;
   - (b) whether the competition's graph tools actually help Gemma 4 E2B/E4B/12B over BM25, dense retrieval and Agentless prompting, especially on unseen codebases;
   - (c) the accuracy/tokens/latency Pareto front on a free Kaggle T4.

The feasibility checks below support this:
- about 210 qualifying instances per month from 8 repositories;
- a working end-to-end call to Gemma 4 E2B on the local GPU.

They also shaped the motivation (§10.5, **corrected 25/10**):
- All four public dev repositories of the main competition (fastapi, rich, requests, httpx) are old, well-known codebases, created between 2011 and 2019.
- The hidden test set comes from private, unfamiliar repositories.
- Codebase familiarity is therefore a confound whenever an agent is tuned on the public repositories, whatever the date of the individual task. Born-after public repositories are the closest public stand-in for the hidden-test condition.
- On 24/10 I claimed the dev set was "essentially pre-cutoff". That was an over-reach: it rested only on our own harvester yielding few post-release instances from those repositories, and the competition curates tasks differently.

---

## 1. Research questions and claims

**RQ1 (resource).** Can we build, automatically and refreshably, a file- and function-level localization benchmark whose instances are provably unseen by Gemma 4?
- "Unseen" is checked at two levels: the issue/fix is post-release, and the codebase is born after the pre-training cutoff.
- The domain is agent/LLM-framework repositories, with a quality and leakage audit.

**RQ2 (memory vs. skill).** How much small-model localization accuracy comes from memory?
- Probe: from the issue text alone, with no repository shown, ask for the file path. This follows the SWE-Bench Illusion probe [13].
- Compare probe accuracy across three cells:
  - old repositories with pre-cutoff issues (T0);
  - old repositories with post-release issues (T2-old);
  - born-after repositories (B).
- Control for guessability with a path-only BM25 ranker, given the same issue and the real file list.
- Also run the probe on the main competition's own repositories (fastapi, rich, requests, httpx), harvested from public GitHub.

**RQ3 (structure).** Does code-graph navigation, using the competition's tool interface, improve small Gemma 4 localization over lexical retrieval, dense retrieval and Agentless-style prompting?
- Does the gain grow on born-after codebases, where priors cannot help?
- Where does it cross over with model size: E2B → E4B → 12B → 26B-A4B?

**RQ4 (budget).** What is the accuracy vs. tokens/latency Pareto front on free hardware?
- Ablations: thinking on/off, context budget (8k/16k/32k) and issue identifier masking.
- Identifier masking tests reliance on lexical cues, the "keyword shortcut" described in [36].

**RQ5 (stretch, gated by D3).** Does a small LoRA trained on pre-release instances (T1) transfer to post-release ones (T2)?

**Claims we expect to support.** These will be pre-registered in the repository before the main runs.
- **C1.** The dataset properties (§4) hold, with a measured gold-noise rate.
- **C2.** Issue-only probe accuracy drops from T0-old to B by more than the path-only guessability control explains. A null result is also reportable and useful.
- **C3.** Graph-guided localization beats Agentless prompting for E2B/E4B on born-after repositories at equal token budget. The hypothesis may fail; we report it either way.
- **C4.** BM25 or dense retrieval followed by a short Gemma rerank dominates the multi-turn agent on the Pareto front for E2B.

---

## 2. Related work (verified, numbered)

Every entry below was checked on its arXiv abstract page, ACL Anthology, the official site or GitHub on 24/10/2026. Entries [13], [27], [28], [29], [30], [31] and [3] were re-checked through the arXiv API in this session.

**Benchmarks for issue resolution**
- **SWE-bench** [1]: 2,294 Python tasks from 12 repositories. **Lite** (300) [2] and **Verified** (500, human-validated) [3] are subsets.
  - OpenAI stopped reporting Verified in Feb 2026, citing contamination and flawed tests [4].
- **SWE-bench Multimodal** [5]: 617 JavaScript tasks.
- Live, decontaminated variants: **SWE-bench-Live** [6] (1,319 tasks from 2024 onward; monthly updates), **SWE-rebench** [7] (21k tasks; rolling-window leaderboard) with **V2** [8] (32k tasks, 20 languages), and **SWE-MERA** [9].
  - **All of them score resolved rate only; none reports localization metrics.**
- **SWE-Bench Pro** [10] and **SWE-Bench Pro Verified** [11] resist contamination through held-out and commercial splits.
- Benchmark validity: **SWE-Bench+** [12] finds the solution leaked in the issue text in 32.67% of successful patches. **The SWE-Bench Illusion** [13] shows file-path recall from the issue alone (up to 76% on Verified vs 53% outside it). **Schrödinger's Code Repository** [14] finds that localization is where de-familiarized repositories hurt most. Wang et al. [15] find many "solved" patches are wrong.
- **Multi-SWE-bench** [16] and **SWE-PolyBench** [17] are multi-language. SWE-PolyBench reports file- and syntax-node-level retrieval precision/recall.
- **SWE-Gym** [18], **SWE-smith** [19] and **R2E-Gym** [20] are training environments.

**Localization benchmarks and methods**
- **Agentless** [21]: hierarchical file → skeleton → line localization. File-level ablation on Lite: prompting 78.7%, embeddings 70.3%, combined 81.7%.
- **LocAgent** with **Loc-Bench** [22]: 560 instances, bugs after Oct 2024, graph-guided agent, Acc@k at file/module/function level.
  - **Loc-Bench's window predates Gemma 4's Jan 2025 cutoff.**
- **OrcaLoca** [23], **CoSIL** [24] and **RepoGraph** [25]: graph search and graph context.
- **SweRank** [26]: retrieve then rerank; a 137M embedder beats Claude-based agents. On Lite, BM25 gets file Acc@1 = 38.7.
- **Long Code Arena** bug localization [37]: about 7.5k pairs, MAP and R@k. **MULocBench** [38]: 1,100 issues, including non-code locations.
- **2026, closest to us:**
  - **IssueLoc-Bench** [27]: Gemma-4-E4B and Qwen3-4B locally; Hit@k, MRR, tokens and time; Verified-derived tasks plus 500 recent issues from 153 repositories.
  - **SemNav** [28]: a Gemma 4 E4B localizer on Lite and PLocBench.
  - **Small Agents with Semantic Search** [29]: sub-2B models trained with SFT and RL; CPU latency.
  - **SWE-Explore** [30], **FuseSearch** [31], **ToolTrain** [32], **LogicLoc** [36].
- Retrieval: **CodeRAG-Bench** [39], **RepoBench** [40], **CoIR** [41], **CoRNStack / CodeRankEmbed** [42], **EmbeddingGemma** [43], **BM25** [44].

**Agent-framework software**
- **AgentIssue-Bench** [33]: 50 reproducible agent-system issues, resolution 0.67–4.67%, no localization.
- Bug taxonomies of agent frameworks [34, 35], with no localization evaluation.

**Gemma**
- Gemma 3 report [45] and **Gemma 4 Technical Report** [46]. The pre-training data cutoff, January 2025, is confirmed in the Hugging Face model cards.
- Time-split methodology follows **LiveCodeBench** [47].

### Novelty positioning

| | Post-Gemma-4 tiers | Born-after-cutoff repos | Agent-framework domain | File + function loc. metrics | Small local Gemma 4 | Competition graph-tool interface | Memorization probe + control |
|---|---|---|---|---|---|---|---|
| SWE-bench-Live [6] / SWE-rebench [7] | partial (rolling dates, not model-keyed) | no | no | no (resolve only) | some | no | no (flags only) |
| Loc-Bench [22] | no (Oct 2024) | no | no | yes | fine-tuned 7B | own graph | no |
| IssueLoc-Bench [27] | partial ("recent") | no | no | file only | yes (E4B) | no | no |
| SemNav [28] | no | no | no | yes | yes (E4B) | own env | no |
| SWE-Bench Illusion [13] | n/a | no | no | file probe | no (frontier) | no | probe, no control |
| **Ours** | **3 tiers keyed to cutoff and release** | **yes** | **yes** | **yes** | **E2B → 26B-A4B on a T4** | **yes (same signatures and schema)** | **yes, with path-only control** |

**What is new:**
1. A model-dated split with a codebase-level control (born-after repositories).
2. The first localization set for agent-framework repositories.
3. Drop-in compatibility with the competition's graph and embedding interface, plus an audit of whether that interface helps small models.
4. A memorization probe with a guessability control, across tiers.

**What is not new**, and we will say so: issue→PR harvesting [6, 7, 37], Acc@k/MRR metrics [22, 26], and running Gemma 4 E4B locally [27, 28].

---

## 3. Domain and sources

**Pool.** Permissively licensed (MIT, Apache-2.0, BSD) Python agent/LLM-framework repositories.

Repository facts checked with `gh api` on 24/10:

| Repo | Created | License | Tier role |
|---|---|---|---|
| openminion/openminion | 2026-06-23 | Apache-2.0 | B + hand-verified slice (our 19 merged PRs) |
| google/adk-python | 2025-04-01 | Apache-2.0 | B; needs a commit linker (Copybara, see §10.3) |
| openai/openai-agents-python | 2025-03-11 | MIT | B |
| strands-agents/harness-sdk | 2025-05-14 | Apache-2.0 | B |
| microsoft/agent-framework | 2025-04-28 | MIT | B (Python part of a mono-repo) |
| a2aproject/a2a-python | 2025-05-09 | Apache-2.0 | B |
| huggingface/smolagents | 2024-12-05 | Apache-2.0 | edge (born 2 months before cutoff) |
| PrefectHQ/fastmcp | 2024-11-30 | Apache-2.0 | old (born just before cutoff) |
| pydantic/pydantic-ai | 2024-06-21 | MIT | old |
| modelcontextprotocol/python-sdk | 2024-09-24 | MIT | old |
| langchain-ai/langgraph, langchain, run-llama/llama_index, crewAIInc/crewAI, stanfordnlp/dspy, browser-use, letta, mem0, camel, haystack, livekit/agents | 2019–2024 | MIT / Apache-2.0 | old |
| fastapi/fastapi, Textualize/rich, encode/httpx (BSD-3), psf/requests | 2011–2019 | MIT / Apache / BSD | old; **the main competition's own repositories**, used as a contamination contrast |

**Excluded:**
- microsoft/autogen: its licence is CC-BY-4.0 at the repository level.
- BerriAI/litellm: NOASSERTION, plus an enterprise folder.
- agno: the repository was renamed; its licence history needs a check first.
- Anything not MIT, Apache-2.0 or BSD.

**Tiers** are based on the issue `created_at` date:
- **T0:** before 2025-02-01.
- **T1:** from 2025-02-01 to 2026-04-01.
- **T2:** from 2026-04-02, the Gemma 4 release. T2 is the primary test tier: post-training data cannot contain issues created after the release.
- **B** is the subset of T1/T2 from repositories created after 2025-01-31.

The Gemma 4 12B checkpoint came out later (2026-06-03). For 12B, the strict tier is issues on or after 2026-06-04, and we report it separately.

---

## 4. Dataset construction

The pipeline is `harvest/harvest.py`, then `harvest/gold.py`. It uses only `gh api`, and all responses are cached and hashed for audit.

1. **Mining.** GitHub GraphQL search lists merged PRs per month window. Issues are linked through `closingIssuesReferences`, with a fallback to closing keywords in the PR body. GitHub only auto-links PRs into the default branch, so repositories that merge into `dev`, like OpenMinion, need the fallback.
2. **Filters**, recorded as a funnel per repository:
   - PR and issue authors are not bots;
   - the issue is created before the PR;
   - at least one modified core `.py` file (not tests, docs, examples, scripts or `.github`);
   - 1–5 core files and at most 400 changed core lines; the file list is not truncated;
   - not trivial: at least 3 changed lines, and no typo/docs/bump title with fewer than 10 lines;
   - at least one test file changed;
   - the issue has at least 80 characters;
   - the issue contains no diff.
3. **Gold.**
   - The landed diff is the merge commit against its first parent, which covers squash and merge commits. Rebase merges fall back to the PR diff at its merge base.
   - Each modified core file is parsed with `ast` at the base commit.
   - Touched old lines and insertion points map to the innermost top-level function, class method or class. Newly added defs are not credited to their neighbours; they go to `added_symbols`.
   - Unit-checked on a synthetic patch: a modified method, an added method and an added line.
   - Gold is reported both as `path::Class.method` and as the dotted id used by the competition graph (`pkg.module.Class.method`).
4. **Leakage and provenance flags:**
   - `issue_mentions_gold_file`, `issue_mentions_gold_function`, `issue_quotes_added_code`;
   - `same_author`: reporter equals fixer, computed but **logins are not stored**;
   - the issue→PR gap in hours.
5. **Primary "independent" subset.** Not `same_author`, a gap of at least 24 h, and no quoted code. In 2026 many issues are filed minutes before the fixing PR, often by the fixer and often with AI assistance. In openai-agents, the median gap is 1.3 h and 46% of PRs open within 1 h. Results are also reported on the full set and on the subset whose issue does not name the gold file.
6. **Hand-verified slice.** The 19 OpenMinion PR→issue pairs (17 issues) in `harvest/openminion_pairs.json`.
   - Each has a reproduction and failing-first regression tests.
   - It is presented as a **case study**, not as the statistics base. We disclose that we authored both the issues and the fixes.
   - The issues were written by an LLM-assisted probe and name many symbols, so this slice is ideal for the identifier-masking ablation.
7. **Audit.**
   - The agent manually reviews 60 random instances against a rubric: gold is the real fix location, no refactor noise, issue is self-contained.
   - We report the gold-noise rate with a 95% CI.
8. **Size target.**
   - **Released:** 1,000+ T2 instances plus T1/T0 for the contrasts. The pool is about 210 a month (§10.3).
   - **Evaluated with Gemma:** a stratified 400. Stratification: 50% born-after; per-repository cap of 60; independent-subset quota of at least 60%. Plus the OpenMinion slice, and 100 T0 + 100 T2 instances from the competition repositories for RQ2.
9. **Snapshots** (implemented 25/10 in `loc/snapshots.py`).
   - Each repository gets one git store. Every base commit is fetched with `--depth=1 --filter=blob:none`, and the work tree is a sparse checkout of `*.py` plus licence files.
   - Blobs arrive lazily at checkout and are deduplicated by git across snapshots: about 3.5 s per new snapshot, and a few MB per extra snapshot of the same repository.
   - Per-instance tarballs would not fit the 13.8 GB free disk. The smoke test used them; they have been deleted.
   - The released Kaggle Dataset ships only content-addressed `.py` files plus each repository's LICENSE files. All repositories are MIT, Apache-2.0 or BSD, which allow redistribution with notice.

**Licence and terms.**
- Code, scripts and harness: Apache-2.0, as the competition rules require.
- Labels and metadata (repository, PR/issue numbers, SHAs, gold paths and symbols, flags): CC BY 4.0.
- Issue text is user content. GitHub's Acceptable Use Policies allow researchers to use "public, non-personal information from the Service for research purposes", provided resulting publications are open access. The Kaggle Writeup becomes CC BY 4.0, so it qualifies.
- We follow the SWE-bench, SWE-rebench and Long Code Arena precedent of shipping issue text, with these safeguards:
  - each record keeps its source URL (attribution);
  - usernames and e-mails are not stored, and @mentions are masked;
  - a rehydration script can refetch text from GitHub instead;
  - a takedown contact is provided.
- Code snapshots keep their upstream licences.
- Competition data, which is under a no-redistribution clause, is **never** redistributed. RQ2 uses our own harvest of the same public repositories.

---

## 5. Tasks and metrics

**Task.** Given the issue text and a repository snapshot at `base_commit`, return a ranked list of files and a ranked list of functions to edit.

**Metrics.**
- **File-level**, for k ∈ {1, 3, 5, 10}:
  - **Acc@k** (all gold in top-k; the LocAgent/SweRank convention);
  - **Hit@k** (any gold in top-k; the IssueLoc/SemNav convention);
  - MRR and NDCG@k.
- **Function-level:** the same metrics at k ∈ {5, 10}. Gold is the set of modified pre-existing functions and methods.
- **Name normalization:** a bare method name matches `Class.method` only if it is unique in that file. The smoke test showed this rule is needed.
- **Cost:** prompt, completion and reasoning tokens; wall-clock seconds per instance; tool calls; tool-call parse failures; peak VRAM.

**Statistics.**
- Bootstrap 95% CIs (1,000 resamples, seed 0), clustered by repository.
- Paired comparisons use McNemar's test (Hit@k) or a paired permutation test (MRR).
- Repository-macro averages are reported next to the micro averages.

---

## 6. Methods (all training-free unless noted)

| Id | Method | Notes |
|---|---|---|
| R0 | Random file; path-only BM25 (issue vs. file paths) | Guessability floor and the control for RQ2 |
| B1 | BM25 [44] over path + content (files) and AST units (functions) | Dependency-free implementation in `loc/bm25.py` |
| B2 | Dense: **EmbeddingGemma 2** [48] (270M text backbone, Apache-2.0, truncated to 256-d MRL, the competition's embedding width) and **jina-embeddings-v2-base-code** [49] (161M, 768-d). Both are official ggml-org GGUFs served through LM Studio / llama.cpp | Changed 25/10 (see §13). Vectors cached by content hash (`data/cache/emb.sqlite`); files ranked by their best unit |
| RM | Repo-map: personalised PageRank over our code graph, seeded by code-like identifiers in the issue. No LLM, CPU only. Inspired by aider's repo-map (https://aider.chat/docs/repomap.html) | Added 25/10. Isolates what the graph alone contributes |
| P | Probe: issue only, no repository shown → 5 paths [13] | RQ2 |
| A | Agentless-style hierarchy [21]: file list → 5 files → skeletons of the top 3 → 10 functions | Two LLM calls |
| G1 | Graph rerank (non-agentic): B1/B2 seeds → 1-hop `calls/imports/contains` neighbours → Gemma listwise rerank | Robust when small models fail at tool calls |
| G2 | Graph agent: ReAct loop of at most 12 steps with **exactly** the competition tools `get_code_neighbors(node, edge_type, max_neighbors)`, `search_similar_code(query, k)`, `get_code_subgraph(nodes)` plus `read_file` | As in LocAgent [22] and CoSIL [24], but with the competition interface |
| L | (stretch) LoRA on E2B or E4B: Agentless step-1 format, trained on T1 and tested on T2, plus a shuffled-label control | RQ5 |
| MS | (optional, only if budget allows) Bash-only agent with read-only commands at the same step budget as G2, contrasting shell navigation with graph tools. It would use mini-swe-agent==2.4.6 with litellm pinned to 1.81.10 and Gemma through Ollama | Execution-based agents follow CodeAct [50]; production agent scaffolds such as the OpenHands SDK [51] |

**Graph builder** (implemented 25/10: `graph/build_graph.py` v0.2, `graph/tools.py`, tests in `tests/test_graph.py`):
- Built with Python `ast`. One pass per file produces a record cached by git blob id; names are resolved per snapshot. About 5 s for a 640-file snapshot on a cold cache.
- Module ids: the path after the last `src/` directory, else the contiguous `__init__.py` package chain. Colliding ids (such as several `conftest.py` files outside packages) fall back to the full path.
- Node ids are dotted fully qualified symbols (module, class, function), each with its source `text`.
- Edges: `contains`, `imports`, `calls` (name-resolved through local scope and imports; unresolved calls are counted and dropped) and `inherits`.
- Output is NetworkX node-link JSON (directed multigraph) with the published fields `directed`, `multigraph`, `graph`, `nodes[id, name, text]` and `edges[source, target, type, key]`.
- **Compatibility, checked locally on 25/10** against one official graph after joining the main track. Details stay in `competition_local/NOTES.md`, which is never published.
  - **Same:** the JSON layout, and the dotted id convention for library code (`src/` stripped, `pkg.module.Class.method`).
  - **Different, by our choice:**
    - we type edges (calls, contains, imports, inherits); the official files use one undifferentiated call relation;
    - we add module nodes;
    - we roll nested functions into their parent;
    - our class text is the full source, not a skeleton.
  - Our graphs load into the same NetworkX structures. A converter to the exact official conventions exists only locally, because rules 3.5.d and 3.6 restrict sharing it (§12b).
  - Our tools (`graph/tools.py`) are an independent implementation with our own JSON format: edge type `edge`, node type `kind`.

---

## 7. Models and ablations

**Models.**
- All instruction-tuned QAT checkpoints from Google, run as q4_0 GGUF through llama.cpp (LM Studio locally, Ollama on Kaggle).
- Temperature 0, seed 0; thinking **off** by default (`reasoning_effort: "none"`).
- A 50-instance subset is repeated 3 times to measure non-determinism.

| Model | Role |
|---|---|
| Gemma 4 E2B | All methods, full evaluated set (local GPU) |
| Gemma 4 E4B | All methods, full evaluated set (Kaggle T4) |
| Gemma 4 12B | All methods, full evaluated set (Kaggle T4), with a strict post-12B-release tier |
| Gemma 4 26B-A4B | A, G1 and G2 on a 150-instance subset (2×T4) |
| Gemma 4 31B | Ceiling on 100 instances only if quota allows (2×T4, tight) |

**Ablations**, each run on the evaluated set unless noted:
1. Graph vs. no graph: G1/G2 against A/B1, at matched token budget.
2. Context budget: 8k / 16k / 32k. The file-list truncation follows B1 rank.
3. Thinking on/off, on E4B and 12B.
4. Issue identifier masking: backticked identifiers, paths, and tokens matching any repository symbol become `[SYMBOL]`.
5. Model size: E2B → 26B-A4B.
6. Tier contrast: T0-old / T2-old / B, for the probe (P) and for A.
7. (stretch) LoRA.

**Main-track compatibility note.** The competition scores only the `-qat-w4a16-ct` (compressed-tensors) checkpoints on L4×4. We use the q4_0 GGUF from the same Gemma 4 QAT family because it runs on a T4 and the GTX 1660S, and we state this difference.

---

## 8. Compute plan

**Local.**
- Hardware: GTX 1660 SUPER, 6 GB VRAM (about 4.9 GB free with the desktop running); 16 GB RAM (about 4–5.7 GB free); 13.8 GB free disk.
- Gemma 4 E2B QAT q4_0 (3.35 GB) fits fully on the GPU: measured 3.1 GB VRAM with a 16k context.
- Measured cost: about 16–25 s and 5.6–7.6k tokens per instance for A.
- Estimated E2B time:
  - 400 instances × (P + A + G1): about 6–8 h overnight;
  - G2 adds about 8–12 h.
- BM25 and graph building run on CPU in minutes.
- E4B (5.15 GB GGUF) does not fit, so it is not run locally.

**Kaggle.**
- Quota: T4×2 (2×15 GB, SM75, no bf16) or P100 (16 GB). About **30 GPU-hours a week** is the community-reported figure and must be checked on the account page. Sessions run up to 9–12 h, and internet is allowed outside the L4 machines.
- Ollama is installed in the notebook and pulls `gemma4:e4b-it-qat`, `12b-it-qat`, `26b-a4b-it-qat` and `31b-it-qat`; all tags were checked on ollama.com. Our client `loc/llm.py` (OpenAI-compatible, stdlib only) works unchanged.
- Fit on a T4:
  - E4B and 12B fit on one T4;
  - 26B-A4B (15.6 GB) and 31B (18.9 GB) need both T4s with a limited KV cache.
- Embeddings: about 1 h on a T4.

Budget to 10/11, about 60–70 GPU-h:

| Job | GPU-h |
|---|---|
| E4B (all methods) | ~10 |
| 12B | ~20 |
| 26B-A4B subset | ~8 |
| Embeddings | ~1 |
| LoRA (stretch) | ~4 |
| Reserve | ~15 |

**Linux runner** (GitHub Actions): CPU-heavy jobs such as full harvesting with clones and graph building, so they don't load the laptop.

---

## 9. Verifiability plan

- **Code.** A public GitHub repository under Apache-2.0, created only after the author approves. It holds the harvester, gold extraction, graph builder, harness, analysis and tests (gold mapping, metrics, path classification, graph schema).
- **Kaggle Dataset.** Instances JSONL, graphs, embeddings, snapshot manifest, model outputs, plus a datasheet with the funnel and audit numbers.
- **Public notebook**, with two modes:
  1. **CPU replay** rebuilds every table and figure from cached model outputs, and needs no GPU.
  2. **GPU mode** re-runs a 50-instance subset end to end with E4B.
- **Exactness.**
  - Model ids are pinned: Kaggle `google/gemma-4/gguf/gemma-4-e4b-it-qat-q4_0-gguf/2` and the like, Hugging Face `google/gemma-4-E2B-it-qat-q4_0-gguf`, and Ollama tags with digests.
  - The local E2B GGUF sha256 is `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`, matching the Hugging Face LFS oid.
  - Seeds, temperature, verbatim prompt templates (appendix), llama.cpp/Ollama versions and the harvest date and queries are all recorded. Raw GitHub API responses are cached and hashed.
- **Pre-registration.** `PREREG.md` (hypotheses C1–C4 plus the analysis plan) is committed before the main Gemma runs. Its commit hash goes in the paper.
- **Before any public push:** `security-preflight.ps1`, and the English preflight (writeup, README, notebook).

---

## 10. Feasibility results (run on 24/10/2026)

### 10.1 Gemma 4 models

Sources: Hugging Face model cards and API, the Kaggle Models API, and the Gemma 4 Technical Report [46].

| Model | Params | Layers | Context | Modalities | HF id | Kaggle Models (Transformers) |
|---|---|---|---|---|---|---|
| E2B | 2.3B effective (5.1B with PLE embeddings) | 35 | 128K | text, image, audio, video | `google/gemma-4-E2B-it` | `google/gemma-4/transformers/gemma-4-e2b-it/1` |
| E4B | 4.5B effective (8B with embeddings) | 42 | 128K | text, image, audio, video | `google/gemma-4-E4B-it` | `.../gemma-4-e4b-it/1` |
| 12B "Unified" | 11.95B (encoder-free multimodal) | 48 | 256K | text, image, audio | `google/gemma-4-12B-it` | `.../gemma-4-12b-it/2` |
| 26B A4B (MoE) | 25.2B total, 3.8B active | 30 | 256K | text, image, video | `google/gemma-4-26B-A4B-it` | `.../gemma-4-26b-a4b-it/1` |
| 31B dense | 30.7B | 60 | 256K | text, image, video | `google/gemma-4-31B-it` | `.../gemma-4-31b-it/1` |

**Dates.**
- Released 2 Apr 2026. Kaggle model instances were created 2026-04-02; the release page says 31 Mar.
- The 12B followed on 3 Jun 2026.
- Pre-training data cutoff: **January 2025** (model cards).

**Licence.** Apache-2.0, and the Hugging Face repositories are not gated.

**Published scores.** LiveCodeBench v6: 31B 80.0, 26B 77.1, 12B 72.0, E4B 52.0, E2B 44.0. Tau2: 76.9 / 68.2 / 69.0 / 42.2 / 24.5. No SWE-bench number is published.

**Quantized variants.**
- Kaggle GGUF instances `gemma-4-{e2b,e4b,12b,26b-a4b,31b}-it-qat-q4_0-gguf`: total sizes 4.3 / 6.1 / 7.2 / 15.6 / 18.9 GB, including the mmproj file. The E2B text weights alone are 3.35 GB.
- `-qat-w4a16-ct` variants are the main track's supported set.
- Thinking: Gemma 4 has configurable thinking, and LM Studio enables it by default. `reasoning_effort: "none"` turns it off; with thinking on, 160 reasoning tokens were spent on a trivial prompt.

**Hardware fit.**

| | Local GTX 1660S (6 GB) | Kaggle T4 (15 GB) | Kaggle 2×T4 | P100 (16 GB) |
|---|---|---|---|---|
| E2B q4 | **yes (measured)** | yes | yes | yes |
| E4B q4 | no (5.15 GB plus KV) | yes | yes | yes |
| 12B q4 | no | yes | yes | yes |
| 26B-A4B q4 | no | no | tight | no |
| 31B q4 | no | no | tight | no |
| E4B fp16 (Transformers) | no | no | yes (16 GB split) | no |

### 10.2 Kaggle CLI

- Authenticated: `kaggle competitions list -s gemma` works and shows `gemma-4-developer-agent-paper` with `userHasEntered=True`. CLI version 2.2.4; 2.2.5 is available.
- Existing setup: `shared/cloud/kaggle-llm` has a private GPU script-kernel template (`kernel-metadata.json`, plus `run_small_llm.py` using Transformers in fp16). It can be reused for the T4 runs once private pushes are approved.
- **Main track (gemma-4-developer-agent).**
  - **Entry deadline 25 Nov 2026.** The scouting note's "26/10" was wrong.

### 10.3 Harvester dry run

Window: post-release, 2026-04-02 → 2026-10-23 (6.7 months). Output is in `data/dryrun/*.funnel.json`.

| Repo | Merged PRs | Issue-linked | **Qualifying** | Per month |
|---|---|---|---|---|
| microsoft/agent-framework (B) | 1,932 | 961 | **407** | 60.7 |
| pydantic/pydantic-ai | 2,513 | 762 | **428** | 63.9 |
| PrefectHQ/fastmcp | 949 | 398 | **270** | 40.3 |
| openai/openai-agents-python (B) | 1,087 | 231 | **167** | 24.9 |
| strands-agents/harness-sdk (B) | 1,375 | 285 | **115** | 17.2 |
| langchain-ai/langgraph | 536 | 23 | 13 | 1.9 |
| google/adk-python (B) | 144 | 9 | 5 | 0.7 |
| huggingface/smolagents | 25 | 4 | 1 | 0.1 |
| **Total (8 repos)** | | | **1,406** | **~210** |
| openminion/openminion (B) | 339 | 10 | 6 generic / **19 curated** | n/a |

Notes:
- **google/adk-python** is developed through Copybara: issues are closed by direct commits, not merged PRs. A commit linker (a `Fixes #N` search plus issue timeline "closed by commit" events) would unlock it.
- **smolagents** has low activity in 2026.
- **langgraph** rarely links issues.

**Gold extraction** on 66 sampled instances:
- 0 parse errors; 100% resolved through merge commit vs. first parent.
- On average 1.56 gold files and 2.8 gold functions per instance; 6% have module-level-only changes.
- Leakage: the issue names a gold file in 38%, a gold function in 62%, and quotes the added code in 3 of 66.
- Reporter equals fixer in 41–58% of candidates in the large repositories.

**Additional runs** (T1 window for openai-agents; T2 for the competition repositories): see §10.5.

### 10.4 End-to-end smoke test (Gemma 4 E2B, local)

Setup: LM Studio (`lms` CLI) with its llama.cpp CUDA12 runtime 2.64.0, `gemma-4-e2b-it-qat` (q4_0 GGUF), 16k context, full GPU offload, thinking off. Script: `loc/smoke.py`.

Three T2 instances. Two of them (openai-agents-4330, fastmcp-5539) have an issue that names neither the gold file nor the gold function. strands-3423 is from a born-after repository.

| Instance (core files) | BM25 file Hit@1 / MRR | Probe (no repo) file Hit@1 | Agentless file Hit@1 / Hit@5 | Agentless function Hit@5 | Agentless tokens (in/out) and time |
|---|---|---|---|---|---|
| openai-agents-4330 (300) | 1 / 1.00 | 1 | 1 / 1 | 0 (name normalization) | 5.6k / 127, 16 s |
| fastmcp-5539 (288) | 0 / 0.11 | 0 | 0 / 0 | 0 | 6.9k / 112, 20 s |
| strands-3423 (286) | 1 / 1.00 | 1 | 0 / 1 | 1 | 7.6k / 120, 22 s |

**Takeaways.**
1. The pipeline works end to end: harvest → gold → snapshot → BM25 → Gemma 4 → metrics.
2. Thinking must be switched off explicitly, or the output budget goes to reasoning and comes back empty. This was found and fixed.
3. The issue-only probe succeeds when the issue names the integration ("litellm", "llamacpp"). **RQ2 needs the guessability control** (path-only BM25), which is now part of the design.
4. Function names need normalization: the model wrote `get_response` for `LitellmModel.get_response`.
5. BM25 is a strong baseline even here, as SweRank and Agentless also report.

### 10.5 T1 tier and competition-repository yield

| Run | Window | Merged PRs | Issue-linked | Qualifying |
|---|---|---|---|---|
| openai/openai-agents-python, **T1** | 2025-03-11 → 2026-04-01 (12.7 months) | 1,017 | 199 | **127** (~10/month) |
| fastapi/fastapi, T2 | 2026-04-02 → 2026-10-23 | 496 (178 by bots) | 2 | **0** |
| Textualize/rich, T2 | same | 10 | 5 (all with pre-window issues) | **0** |
| encode/httpx, T2 | same | **0** | 0 | **0** |
| psf/requests, T2 | same | 67 | 8 | **1** |

**Finding that shapes the motivation** (corrected 25/10). With our filters (an issue linked to a PR that also changes tests), the main competition's four public repositories are mature or dormant.
- They yield almost no post-release instances: 1 in 6.7 months.
- **Correction:** this says nothing about the dates of the competition's own dev tasks, which are curated differently (for example from PR descriptions). The 24/10 inference that the dev set is "pre-cutoff" was wrong and is withdrawn.
- The argument that stands: every public dev repository is an old codebase (2011–2019) whose code Gemma 4 has very likely seen, while the hidden test set uses **private repositories**.
- So participants tune on familiar code and are tested on unfamiliar code. Born-after-cutoff public repositories are the closest public stand-in for the hidden-test condition, and that is the main relevance argument for the resource.
- RQ2's "old repository" cells will use T0 instances harvested from these four repositories, plus pydantic-ai, fastmcp and langgraph for post-release old-repository instances.

T1 yield is lower than T2: about 10 a month against 25 for openai-agents. Across the pool it still gives about 1,000+ T1 instances, enough for the RQ5 LoRA training set and for the tier contrasts.

---

## 11. Timeline, milestones, decision points

| Date | Milestone |
|---|---|
| Sat 24/10 | ✓ Design, feasibility, dry run, end-to-end smoke test |
| Sun 25/10 – Tue 27/10 | **M1 Harvest v0.1**: 15+ repos, tiers T0/T1/T2, gold, flags, adk commit linker, OpenMinion slice; 60-instance audit. **D1 (27/10):** if the independent T2 subset has fewer than 250 instances, widen the repo list |
| Wed 28/10 – Thu 29/10 | **M2** Graph builder (competition schema), embeddings, R0/B1/B2 on the full set, name normalizer, metrics and bootstrap, tests. Write `PREREG.md` and commit it |
| Fri 30/10 – Mon 2/11 | **M3 Gemma runs**: E2B local (full); E4B and 12B on Kaggle (50-instance pilot, then full); probe tiers; masking; context and thinking ablations. **D2 (31/10):** if G2 tool-call validity is below 80% on E4B, make G1 the main graph method and report G2 as exploratory |
| Tue 3/11 | 26B-A4B subset. **D3:** LoRA goes ahead only if M3 is complete and at least 15 GPU-h remain |
| Wed 4/11 – Fri 6/11 | Analysis, figures, writeup draft (≤ 3,000 words) |
| Sat 7/11 – Sun 8/11 | Polish, notebook replay from a clean environment, security and English preflights |
| Mon 9/11 | User review; approval to publish the repo, dataset and notebook |
| Tue 10/11 | **Submit the writeup** (internal target). Buffer to 12/11 23:59 UTC |

**Calendar conflicts:**
- TastePay closes on 31/10.
- OpenMinion ETAs fall on 28/10 – 3/11 (#388–#392, #369 on 6/11).

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Novelty overlap with IssueLoc-Bench, SemNav and Loc-Bench | Position on the born-after codebase control, the agent-framework domain, the competition-interface audit and the controlled probe. Never claim "first post-cutoff" or "first Gemma 4 localization" |
| Small models fail at tool calling (G2) | G1, the non-agentic graph rerank; report tool-call validity (D2) |
| Gold noise (refactors, several valid fixes) | ≤5 files and ≤400 lines; the audit; Hit@k beside Acc@k |
| Issue leakage and AI-written "self-issues" | Independent subset as primary; mention flags; masking ablation |
| OpenMinion slice is self-authored | Case study only, disclosed |
| Kaggle quota or T4 limits | GGUF through Ollama; E2B local; subsets for 26B/31B; Linux runner for CPU work |
| Disk (13.8 GB free) | Blobless clones plus `cat-file`; main-track data only on Kaggle |
| Licence of issue text | CC BY 4.0 labels, attribution, no logins, rehydration script, takedown contact |
| Time conflicts | Decision points D1–D3; LoRA is optional |


## 13. Steps 1–2 results (25/10/2026)

### 13.1 Dataset v0.1

- **Harvest:** 30 runs over 26 repositories (`harvest/repos_v01.json`), 0 errors, 4,795 qualifying candidates.
  - Old repositories' T0 and T1 candidates are capped at 60 per run, giving **4,097 instances** with file-level gold.
  - By tier: T0 282 · T1 1,390 · **T2 2,425**.
  - **1,790 born-after.** 1,765 independent, of which 790 are T2.
  - Gold size: on average 1.61 files and 3.14 functions per instance; 112 instances have no function-level gold.
- **New since the design:**
  - a **commit linker** for Copybara repositories: adk-python yields 498 instances that PR mode cannot see;
  - the curated OpenMinion slice (19);
  - flags `mostly_new_code` (111) and `issue_older_than_1y` (69), both removed from the evaluation strata.
- **Leakage:**
  - the issue names a gold file in 41% of instances and a gold function in 60%;
  - it quotes the added code in 2.7%;
  - the reporter is also the fixer in 36%.
- **Gold audit** (AUDIT.md): **57/60 gold-correct (95.0%, Wilson CI 86.3–98.3%)**, and 55% of issues are prescriptive.
  - The audit found 4 extractor errors. Each was fixed with a regression test, and gold was regenerated from cached data before any model run.
  - One linking error is excluded.
- **Evaluation set** (`harvest/select_eval.py --seed 0`): 519 ids.
  - `main`: 400 T2, 200 born-after, 328 independent;
  - `om`: 19;
  - `rq2_t0`: 100.
- **Release files** are built locally under `data/v0.1/release/`: a labels file without issue text, an issues file with masking, and stats. Nothing has been published.

### 13.2 Retrieval baselines (no LLM)

Results are in `data/v0.1/results/baselines.md`: micro mean, with a repository-clustered bootstrap 95% CI.

| Method | Subset | n | File Hit@1 | File Hit@5 | File MRR | Function Hit@10 |
|---|---|---|---|---|---|---|
| B1 BM25 | all | 518 | 0.556 | 0.799 [0.74, 0.85] | 0.670 | 0.713 |
| B1 BM25 | **independent ∩ T2 (primary)** | 328 | 0.637 | **0.854** [0.80, 0.90] | 0.734 | 0.772 |
| B1 BM25 | born-after | 219 | 0.571 | 0.808 | 0.677 | 0.749 |
| B1 BM25 | old repository, T2 | 200 | 0.645 | 0.865 | 0.749 | 0.794 |
| B1 BM25 | T0 (old repositories) | 99 | 0.343 | 0.646 | 0.496 | 0.469 |
| B1 BM25 | issue names no gold file | 257 | 0.448 | 0.728 | 0.577 | 0.630 |
| B1 BM25 | OpenMinion | 19 | 0.105 | 0.474 | 0.299 | 0.421 |
| R0-path (guessability) | all | 518 | 0.266 | 0.585 | 0.411 | – |
| R0-random | all | 518 | 0.019 | 0.054 | 0.043 | – |

**Readings** (descriptive; the pre-registered tests run in step 3):
- **BM25 is a strong floor**, as SweRank [26] and Agentless [21] also report. Any Gemma method has to beat file Hit@5 ≈ 0.80–0.85.
- **Path names alone are very informative:** R0-path reaches Hit@5 0.585. The probe P in RQ2 must therefore be read against this control, not against random.
- **Old repositories are easier than born-after ones for path-only and BM25 alike:** R0-path 0.645 versus 0.566. This is a first hint of the familiarity and naming confound that RQ2 targets.
- **T0 is harder than T2 for BM25** (0.646 versus 0.835). T0 is dominated by older, terser issues from the competition and old agent repositories. Tier and repository are confounded here, so the RQ2 analysis compares within repository groups.
- **OpenMinion's own issues are hard for BM25** (deep package paths, behaviour-level descriptions). That makes the slice an informative case study.

### 13.3 Graph builder and tools

- **Status:** `graph/build_graph.py` v0.3 (cached per blob, about 5 s for a 640-file snapshot on a cold cache) and `graph/tools.py`, with the same three tool signatures as the competition. Covered by tests.
- **Schema compatibility:** checked locally against one official graph (§6; details private in `competition_local/NOTES.md`). The layout and id convention match. The differences are deliberate.
- **Fix found by the comparison:** definitions inside module-level `if`/`try`/`with` blocks were ignored. All four consumers (gold, BM25 units, graph, skeletons) now share `harvest/pyast.py`.

### 13.4 Dense retrieval and graph reachability (completed 25/10)

**B2: EmbeddingGemma 2.**
- Run in the private Kaggle kernel `agentloc-b2-dense`: T4, float32, 256-d MRL, max_seq_length 512.
- Environment: torch 2.11, transformers 5.20, sentence-transformers 6.2.
- Scored on the same 518 instances as B1 (paired):

| Subset | B1 BM25: file Hit@5 / function Hit@10 | B2 dense: file Hit@5 / function Hit@10 |
|---|---|---|
| all | **0.799** / **0.713** | 0.743 / 0.516 |
| independent ∩ T2 | **0.854** / **0.772** | 0.750 / 0.505 |
| T0 (old repositories, older issues) | 0.646 / 0.469 | **0.727** / **0.520** |
| issue names no gold file | 0.728 / 0.630 | 0.712 / 0.496 |

Dense retrieval loses to BM25 overall, because so many issues carry lexical cues (55% are prescriptive). It **wins on T0** and nearly ties when the issue names no gold file. A free-text query encoder therefore helps exactly where lexical shortcuts are missing. This is the motivation for the hybrid seeds in G1, and for testing a query encoder next to the provided symbol-keyed similarity tool.

**Graph reachability** (local, 512 instances with function gold; `data/v0.1/results/graph_reachability_summary.json`):
- Every gold function is a node of our graph, and 30.8% of call sites resolve statically.
- Seeds are the B1 top-10 functions. Gold functions reached:
  - 0 hops: 38.6% (71.3% of instances have at least one);
  - **1 hop: 58.2% (81.2%)**, median candidate pool 59;
  - 2 hops: 77.3% (89.3%), median pool 307.
- One hop raises the recall ceiling by about 20 points at a pool size a single Gemma rerank can read (G1 caps it at 40). This is the mechanism H2 tests.

### 13.4c No-LLM baselines on the evaluation set (paired, n = 519) and the full set

Results are in `data/v0.1/results/eval_retrieval.md` and `full_retrieval.md`. RM ran in the private CPU kernel `agentloc-rm`; full-set R0/B1 ran in `agentloc-full-baselines` (4,097 instances, about 2.7 h of CPU).

| Method | File Hit@1 | File Hit@5 | Function Hit@10 | File Hit@5, independent ∩ T2 | File Hit@5, issue names no gold file |
|---|---|---|---|---|---|
| R0-random | 0.019 | 0.054 | – | 0.037 | 0.039 |
| R0-path | 0.266 | 0.586 | – | 0.595 | 0.519 |
| **RM** (repo-map PPR, no LLM) | 0.484 | **0.800** | 0.635 | 0.845 | 0.686 |
| B1 BM25 | **0.557** | **0.800** | **0.714** | **0.854** | **0.729** |
| B2 EmbeddingGemma 2 | 0.437 | 0.744 | 0.517 | 0.750 | 0.713 |

- The graph alone, seeded only by identifiers in the issue, equals BM25 on file Hit@5. It trails BM25 on top-1 and at function level.
- **Full set, B1 (n = 4,097):** file Hit@5 0.793. By tier: T2 0.854 · T1 0.725 · T0 0.610. Born-after 0.790; independent 0.755.

### 13.4d Step 3, interim (25/10): Gemma 4 E2B and E4B on the evaluation set

**Setup.** Private Kaggle T4 kernels `agentloc-gemma-e2b` and `agentloc-gemma-e4b`:
- Ollama 0.40.11 with QAT tags; `num_ctx` 16384; thinking off;
- prompts frozen at commit `cb9763e` after the pilot (PREREG §8).

**Results:** paired, n = 519 (`data/v0.1/results/eval_e2b_e4b.md`).

| Method | File Hit@1 | File Hit@5 | Function Hit@10 | Median prompt tokens | Median s/instance |
|---|---|---|---|---|---|
| B1 BM25 | 0.557 | 0.800 | 0.714 | 0 | – |
| RM (graph, no LLM) | 0.484 | 0.800 | 0.635 | 0 | – |
| P, E2B (issue only) | 0.516 | 0.572 | – | 864 | 1.2 |
| A, E2B (Agentless-style) | 0.613 | 0.805 | 0.563 | 8,096 | 5.1 |
| **G1, E2B** (graph rerank) | **0.748** | **0.871** | **0.760** | 3,063 | 1.6 |
| P, E4B | 0.532 | 0.570 | – | 864 | 1.9 |
| A, E4B | 0.653 | 0.813 | 0.614 | 7,938 | 8.5 |
| **G1, E4B** | **0.801** | **0.896** | **0.797** | 3,063 | 2.8 |

**Pre-registered tests so far** (`loc/analysis.py`; 12B, G2 and A-masked still running):
- **H2 supported for both E2B and E4B.** On independent ∩ main (n = 325), function Hit@10 is G1 0.785 vs A 0.551 for E2B and G1 0.840 vs A 0.612 for E4B. Paired permutation test, Holm-corrected p = 0.0002 for both.
  - The second clause, a larger gain on born-after repositories, is **not supported**: interaction −0.048 for both models (CI around 0).
- **H3** (descriptive): G1 − A = 0.210 [0.141, 0.289] for E2B and 0.203 [0.150, 0.257] for E4B. 12B is pending (final in §13.4e).
- **H1 not supported** for E2B or E4B. The diff-of-diffs is −0.115 and −0.125, and both CIs include 0. The probe's apparent success comes from file names stated in the issue, which T2 issues contain more often than T0 issues.
- **H5, B1 part:** masking costs 0.39–0.41 file Hit@5 both with and without a file mention (interaction −0.01, CI includes 0). The A-e4b part is pending (final in §13.4e).

**Exploratory, not pre-registered** (labelled as such in the paper). On T2 issues that name no gold file, the issue-only probe (E2B) scores:

| Repositories | n | Probe file Hit@5 | Path-only control Hit@5 |
|---|---|---|---|
| Old | 86 | 0.384 | 0.605 |
| Born-after | 91 | 0.176 | 0.538 |

With issues of the same period, the model guesses files twice as well in familiar codebases (+0.21), while the path control differs much less (+0.07). This is the codebase-familiarity effect that the born-after control was designed to expose. The pre-registered T0-vs-born-after contrast mixes in older, terser issues and cannot show it.

### 13.4e Step 3, update (26/10): 12B and G2

The full numbers are in `WRITEUP.md` §5 and `data/v0.1/results/{eval_12b,g2_subsample}.md`.
- **12B** (digest 38044be4…): G1 reaches file Hit@1 0.827, file Hit@5 0.911 and function Hit@10 0.811. A reaches 0.740, 0.877 and 0.706.
- **G2** (200-instance subsample): function Hit@10 is 0.365 for E2B and 0.465 for E4B, against 0.805 and 0.825 for G1. G2 uses 27–34k median tokens; 86% (E2B) and 97% (E4B) of its tool calls are valid.
- **Hypotheses:**
  - H1 is not supported for any model (12B −0.06 [−0.21, 0.11]).
  - H3 shows the predicted monotonic shrink: 0.210, 0.203, 0.129.
  - H4 is supported: G1 dominates G2.
- **Exploratory familiarity gap** (no-mention T2 issues, probe P): +0.21 for E2B, +0.16 for E4B and +0.18 for 12B, against +0.07 for the path control.
- **Final, 26/10 (all step-3 kernels collected).**
  - **H5, A part** (`agentloc-gemma-amask-e4b`, `main`, n = 400): masking cuts A-e4b file Hit@5 from 0.825 to 0.422. The drop is 0.46 when the issue names the gold file and 0.33 when it does not; the interaction is 0.134 [−0.028, 0.282], so H5 is **not supported** for A either. BM25 falls from 0.853 to 0.453, with drops of 0.39 and 0.41 (interaction −0.012 [−0.127, 0.098]).
  - **G2 with 12B** (`agentloc-gemma-g2-12b`, digest 38044be4…, 3.7 GPU-hours of model time): function Hit@10 0.355 on the 200-instance subsample, below E4B (0.465) and far below G1-12b (0.840). Its tool calls are 99% valid and it explores longer (median 11 tool calls), but 98 of 200 runs reach the 12-step cap without naming a function, at a median of 99.1k prompt tokens. On the 102 runs that answered, function Hit@10 is 0.696 against 0.843 for G1-12b on the same instances (exploratory, not pre-registered). H4 is defined for E2B only and is unchanged; re-running `loc/analysis.py` with this file leaves `hypotheses.json` byte-identical.
  - **Final pre-registered verdicts:** H1 not supported (all three models); H2 supported for the main clause (G1 > A, Holm p = 0.0002 for E2B and E4B), not for the born-after interaction; H3 holds descriptively (0.210, 0.203, 0.129); H4 supported (G1 dominates G2 for E2B); H5 not supported (B1 and A).
  - **GPU quota** after all runs: 23.64 h used, 6.36 h left until the 31/10 reset.
- **Optional, not pre-registered as a test: Gemma 4 26B-A4B** (`gemma4:26b-a4b-it-qat`: 14.4 GB of weights plus a 1.2 GB projector, so Ollama splits it over both T4s).
  - Config `kaggle/configs/gemma-26b.json`: P, A and G1 on 150 ids drawn from the G2 subsample (75 born-after, 75 old; `harvest/select_subsample.py --seed 0`), with a 6 h budget. The ids travel inline in the config, so the private dataset needs no new version.
  - Push after the quota reset on 31/10 00:00 UTC: `python -I kaggle/push.py gemma_eval --gpu --kernel-source agentloc-b2-dense --config-file kaggle/configs/gemma-26b.json --push`.
  - Results will be labelled optional and descriptive (one more point on the H3 size trend, paired on the same 150 instances) and never pooled into the pre-registered tests.

## References

1. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR 2024. https://arxiv.org/abs/2310.06770
2. SWE-bench Lite. https://www.swebench.com/lite.html
3. OpenAI, *Introducing SWE-bench Verified*, 2024. https://openai.com/index/introducing-swe-bench-verified/
4. OpenAI, *Why we no longer evaluate SWE-bench Verified*, Feb 2026. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ (the page returns 403 to the fetcher; content taken from search snippets)
5. Yang et al., *SWE-bench Multimodal*, ICLR 2025. https://arxiv.org/abs/2410.03859
6. Zhang et al., *SWE-bench Goes Live!*, 2025. https://arxiv.org/abs/2505.23419
7. Badertdinov et al., *SWE-rebench*, NeurIPS 2025 D&B. https://arxiv.org/abs/2505.20411
8. Badertdinov et al., *SWE-rebench V2*, 2026. https://arxiv.org/abs/2602.23866
9. Adamenko et al., *SWE-MERA*, EMNLP 2025 Demos. https://arxiv.org/abs/2507.11059
10. Deng et al., *SWE-Bench Pro*, 2025. https://arxiv.org/abs/2509.16941
11. Zheng et al., *SWE-Bench Pro Verified*, 2026. https://arxiv.org/abs/2609.08149
12. Aleithan et al., *SWE-Bench+*, 2024. https://arxiv.org/abs/2410.06992
13. Liang et al., *The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason*, 2025. https://arxiv.org/abs/2506.12286
14. Chen et al., *Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?*, 2026. https://arxiv.org/abs/2609.27891
15. Wang et al., *Are "Solved Issues" in SWE-bench Really Solved Correctly?*, ICSE 2026. https://arxiv.org/abs/2503.15223
16. Zan et al., *Multi-SWE-bench*, 2025. https://arxiv.org/abs/2504.02605
17. Rashid et al., *SWE-PolyBench*, 2025. https://arxiv.org/abs/2504.08703
18. Pan et al., *Training Software Engineering Agents and Verifiers with SWE-Gym*, ICML 2025. https://arxiv.org/abs/2412.21139
19. Yang et al., *SWE-smith*, 2025. https://arxiv.org/abs/2504.21798
20. Jain et al., *R2E-Gym*, 2025. https://arxiv.org/abs/2504.07164
21. Xia et al., *Agentless: Demystifying LLM-based Software Engineering Agents*, FSE 2025. https://arxiv.org/abs/2407.01489
22. Chen et al., *LocAgent: Graph-Guided LLM Agents for Code Localization*, ACL 2025. https://arxiv.org/abs/2503.09089
23. Yu et al., *OrcaLoca*, ICML 2025. https://arxiv.org/abs/2502.00350
24. Jiang et al., *CoSIL: Issue Localization via LLM-Driven Iterative Code Graph Searching*, ASE 2025. https://arxiv.org/abs/2503.22424
25. Ouyang et al., *RepoGraph*, ICLR 2025. https://arxiv.org/abs/2410.14684
26. Reddy et al., *SweRank: Software Issue Localization with Code Ranking*, ICLR 2026. https://arxiv.org/abs/2505.07849
27. Al Awad & Ivanov, *Cost-Effective Repository Exploration for Agentic Issue Localization* (IssueLoc-Bench), 2026. https://arxiv.org/abs/2608.29675
28. Wei et al., *Semantic Navigation for Issue Localization in Code Repository* (SemNav), 2026. https://arxiv.org/abs/2609.31176
29. Lasbordes et al., *Small Agents with Semantic Search: Efficient Multilingual Code Localization*, 2026. https://arxiv.org/abs/2610.05099
30. Zhang et al., *SWE-Explore*, 2026. https://arxiv.org/abs/2606.07297
31. Xu et al., *Learning Adaptive Parallel Execution for Efficient Code Localization* (FuseSearch), Findings of ACL 2026. https://arxiv.org/abs/2601.19568
32. Ma et al., *Tool-integrated Reinforcement Learning for Repo Deep Search* (ToolTrain), 2025. https://arxiv.org/abs/2508.03012
33. Rahardja et al., *Can Agents Fix Agent Issues?*, NeurIPS 2025. https://arxiv.org/abs/2505.20749
34. Zhu et al., *Where Agent Frameworks Fall Short*, 2026. https://arxiv.org/abs/2602.21806
35. Islam et al., *When Agents Fail: A Comprehensive Study of Bugs in LLM Agents with Automated Labeling*, ISSRE 2026. https://arxiv.org/abs/2601.15232
36. Xu et al., *Neurosymbolic Repo-level Code Localization* (LogicLoc), 2026. https://arxiv.org/abs/2604.16021
37. Bogomolov et al., *Long Code Arena*, 2024. https://arxiv.org/abs/2406.11612
38. Zhang et al., *A Benchmark for Localizing Code and Non-Code Issues in Software Projects* (MULocBench), 2025. https://arxiv.org/abs/2509.25242
39. Wang et al., *CodeRAG-Bench*, Findings of NAACL 2025. https://arxiv.org/abs/2406.14497
40. Liu et al., *RepoBench*, ICLR 2024. https://arxiv.org/abs/2306.03091
41. Li et al., *CoIR*, ACL 2025. https://arxiv.org/abs/2407.02883
42. Suresh et al., *CoRNStack*, ICLR 2025. https://arxiv.org/abs/2412.01007
43. Schechter Vera et al., *EmbeddingGemma*, 2025. https://arxiv.org/abs/2509.20354
44. Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, FnTIR 3(4), 2009. https://nowpublishers.com/article/Details/INR-019
45. Gemma Team, *Gemma 3 Technical Report*, 2025. https://arxiv.org/abs/2503.19786
46. Gemma Team, *Gemma 4 Technical Report*, 2026. https://arxiv.org/abs/2607.02770; model card https://huggingface.co/google/gemma-4-12B-it
47. Jain et al., *LiveCodeBench*, ICLR 2025. https://arxiv.org/abs/2403.07974
48. Google DeepMind, *EmbeddingGemma 2* model card, Sep 2026 (Apache-2.0). https://huggingface.co/google/embeddinggemma-2; GGUF https://huggingface.co/ggml-org/embeddinggemma-2-GGUF
49. Günther et al., *Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents*, 2023. https://arxiv.org/abs/2310.19923; model https://huggingface.co/jinaai/jina-embeddings-v2-base-code (GGUF https://huggingface.co/ggml-org/jina-embeddings-v2-base-code-Q8_0-GGUF)
50. Wang et al., *Executable Code Actions Elicit Better LLM Agents* (CodeAct), ICML 2024. https://arxiv.org/abs/2402.01030
51. Wang et al., *The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents*, MLSys 2026. https://arxiv.org/abs/2511.03690

Further verified works, to use in the writeup if space allows:
- Agents: SWE-agent (2405.15793), OpenHands (2407.16741), AutoCodeRover (2404.05427), CodeMonkeys (2501.14723), SWE-RL (2502.18449).
- Localization: CoRet (2505.24715), Nemotron-CORTEXA (PMLR v267), RepoNavigator (2512.20957), SpIDER (2512.16956), MM-IssueLoc (2607.15205).
- Bug reports: *What Makes a Good Bug Report for an AI Agent?* (2607.07593).
