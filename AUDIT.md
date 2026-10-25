# Gold audit, dataset v0.1

## Method (fixed before reading any sampled instance)

**Sample.**
- 60 instances from `data/v0.1/gold.jsonl`, drawn by `harvest/audit_view.py --sample 60 --seed 0`.
- Composition: 5 OpenMinion, about 25 from born-after repositories, and the rest from old repositories, picked at random within each group.
- The ids are frozen in `data/v0.1/audit_ids.txt`.

**Annotator.**
- The research agent (Claude), working alone. It wrote the pipeline, which is a known bias. To limit that, each judgement compares the issue text and the landed core-file diff, not the automatic labels.
- Disagreements with the automatic labels are listed one by one below. An optional human spot-check of 20 instances would give a second opinion.

**Card shown per instance** (`audit_view.py`): an issue excerpt (1,100 characters), the fix URL, the commit message, the core-file diff (up to 70 lines per file), the automatic gold files and functions, the added symbols, and the leakage flags.

**Rubric** (one label per question):

| Q | Question | Labels |
|---|---|---|
| G1 | Are the gold files where the reported problem is fixed? | `ok` = every gold file carries part of the fix · `extra` = the fix is there, plus incidental files (refactors, renames, typing, logging) · `wrong` = the diff does not address the issue |
| G2 | Are the gold functions the fix locations? | `ok` · `extra` (incidental functions included) · `missing` (fix code sits in functions the extractor did not credit, e.g. module-level or new helpers) · `wrong` · `n/a` (no function-level gold) |
| G3 | Is the issue a usable localization query? | `ok` = describes a symptom or request · `prescriptive` = names the exact fix location or change ("change X in file Y") · `thin` = too little information to localize without guessing |
| G4 | Type | `bug` · `feature` · `refactor/cleanup` · `docs/config` |
| G5 | Are the automatic mention flags right? | `ok` · or the specific flag that is wrong |

**Decision rules.**
- An instance counts as **gold-correct** when G1 ∈ {ok, extra} and G2 ∈ {ok, extra, n/a}. Hit@k metrics tolerate incidental extra gold; `missing` or `wrong` makes the gold misleading.
- Instances judged G1 = `wrong` go on the exclusion list in PREREG §5, before any Gemma run.
- Rates come with Wilson 95% intervals.

## Results (25/10/2026)

The per-instance labels are in `data/v0.1/audit.jsonl`, which stays local. G2 was judged on the final pipeline, after the extractor fixes listed below.

| Question | Labels (n = 60) |
|---|---|
| G1 gold files | ok 53 · extra 5 · wrong 2 |
| G2 gold functions | ok 50 · extra 6 · n/a 2 · missing 1 · wrong 1 |
| G3 issue as a query | ok 25 · **prescriptive 33** · thin 2 |
| G4 type | bug 45 · feature 11 · docs/config 2 · refactor 1 · n/a 1 |

- **Gold-correct** (G1 ∈ {ok, extra} and G2 ∈ {ok, extra, n/a}): **57/60 = 95.0%**, Wilson 95% CI [86.3%, 98.3%].
  - Among instances eligible for the evaluation strata (excluding the one flagged `mostly_new_code`): **57/59 = 96.6%** [88.5%, 99.1%].
- **Exact** (no incidental extras): 51/60 = 85.0% [73.9%, 91.9%].
- **Prescriptive issues:** **33/60 = 55%** [42.5%, 66.9%].
  - These issues name the exact file or function to change, or state the change itself. Many are self-filed minutes before the fix, and 3 of the 60 say outright that an AI agent wrote them.
  - This supports keeping the `independent` and `no_file_mention` subsets and the identifier-masking ablation (PREREG H5) in the main analysis.
- **Exclusion list** (PREREG §5): `microsoft__agent-framework-9375`, in `data/v0.1/audit_exclusions.txt`.

## Disagreements with automatic labels

**Extractor errors** (all four were fixed, and gold was regenerated from cached API data before any model run):

1. **Appended function with a multi-line signature** (`google__adk-python-cfc4e3d6f60`).
   - A new function added after an existing one was credited to the existing one. The signature's closing `) -> T:` line, at indent 0, ended the "new def" block.
   - Fix: a new def's signature is tracked by bracket balance, and its body by indentation.
   - Over the whole set, regeneration removed 36 wrong function credits.
2. **New members inside existing scopes (rule made explicit).**
   - A new method added to an existing class now credits the class.
   - A nested helper added inside an existing function or method now credits that function.
   - Sibling functions and methods credit nothing.
   - This added 733 class credits and 38 function credits. It is documented as a design rule, not a bug.
3. **Stacked-PR contamination** (`openminion__openminion-351`).
   - The PR file list still contained a file from an earlier stacked PR (#350), so the fallback "PR diff at merge base" was used and the gold included a foreign function.
   - Fix: the landed merge diff is preferred whenever it touches any of the PR's core files. Fallbacks dropped from 11 to 3, and file gold changed for 9 instances.
4. **Re-wrapped `def` lines treated as new defs** (`PrefectHQ__fastmcp-3041`, `deepset-ai__haystack-8176`).
   - A signature removed and re-added over several lines counted as a new member, which spuriously credited the class. It also polluted `added_symbols`.
   - Fix: a `+def` whose name also appears on a `-def` line in the same patch is a modification.
   - This removed 222 spurious credits across 178 instances.

Each fix has a regression test (`tests/test_gold_and_paths.py`, `tests/test_pyast.py`; 28 tests pass).

**Linking and selection errors:**

5. **Linking error** (`microsoft__agent-framework-9375`): the PR's closing reference points to an unrelated issue filed 470 days earlier. It is excluded from evaluation.
   - Old issues are risky in general: 2 of the 60 sampled were over 400 days old, and one of those was this error.
   - A new flag, `issue_older_than_1y`, marks 69 instances (1.7%). They are kept in the release and removed from the evaluation strata.
6. **Mostly-new-code fixes** (`deepset-ai__haystack-8810`): the `mostly_new_code` flag correctly identified that the only "gold" file was an `__init__` export. These instances are already removed from the evaluation strata.

**Kept, with caveats:**

7. **Not representable at function level** (`pydantic__pydantic-ai-7226`, `microsoft__agent-framework-7695`): the fix is a module-level constant. File-level gold is right. Function-level gold is empty or contains only incidental edits. Such cases are reported through `module_level_change_files`.
8. **Parallel or broad fixes** (5 instances labelled `extra`): the PR fixes the reported bug and also a sibling occurrence (Redis next to Valkey, LiteLLM next to OpenAI, OutputAdapter next to ConditionalRouter), or it lands inside a broad feature PR. Hit@k tolerates this; Acc@k (all gold) is stricter, which is one reason both are reported.
9. **Mention flags:** the function-mention flag can fire on common words (e.g. `stream`). It is literally correct by definition, but it is a weak signal. The file-mention flag matched every case checked.

**Limitations.** There was a single annotator, and it is the same agent that wrote the pipeline. Judgements compared the issue text with the landed diff and not with the labels, and every extractor error found was fixed in code rather than relabelled. An independent spot-check by a human annotator (20 ids from `audit_ids.txt`) is still recommended before publication.
