"""Phase 3: derive file- and function-level gold from the landed diff.

For each candidate (harvest.py output) this script:
  1. takes the PR's merge commit and diffs it against its first parent
     (squash and merge commits); if that diff does not cover the PR's core
     files (rebase merges), it falls back to the PR diff at its merge base;
  2. downloads each modified core file at the base commit and parses it with
     Python's `ast`;
  3. maps every removed/changed old line (and every insertion point) to the
     enclosing top-level function, class method or class;
  4. adds leakage flags (issue mentions the gold file / function, or quotes
     the added code).

Usage:  python -I harvest/gold.py data/dryrun/openai__openai-agents-python.candidates.jsonl \
            --limit 20 --seed 0
"""
from __future__ import annotations

import argparse
import ast
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # third-party sources with odd escapes
import json
import random
import re
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh  # noqa: E402
from paths import is_core_py  # noqa: E402
from pyast import methods, span, top_level  # noqa: E402

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DEF_RE = re.compile(r"^\+\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def _brackets(s: str) -> int:
    code = s.split("#")[0]
    return sum(code.count(c) for c in "([{") - sum(code.count(c) for c in ")]}")


def diff_points(patch: str) -> tuple[set[int], list[str], list[tuple[int, int]]]:
    """Parse a unified diff into (touched old lines, added lines, new-def points).

    * removed lines touch their old line; an added line touches old-1 (the line
      it is inserted after);
    * a newly added def/class (with decorators, multi-line signature and body)
      touches nothing by itself; it is returned as a new-def point
      (anchor old line, indent) and credited later only to an enclosing scope
      that it is nested in (see credit_new_defs).
    """
    touched: set[int] = set()
    added: list[str] = []
    new_defs: list[tuple[int, int]] = []
    # a `def` that is removed and re-added (e.g. a re-wrapped signature) is a modification
    removed_names = {m.group(1) for ln in patch.splitlines()
                     if ln.startswith("-") and (m := DEF_RE.match("+" + ln[1:]))}
    old = 0
    block: int | None = None  # indent of a brand-new def/class block
    pending = 0  # open brackets in a new decorator or signature
    sig = None  # None | "decorator" | "def": inside a continuation
    for line in patch.splitlines():
        m = HUNK_RE.match(line)
        if m:
            old, block, pending, sig = int(m.group(1)), None, 0, None
            continue
        if line.startswith("-"):
            touched.add(old)
            old += 1
            block, pending, sig = None, 0, None
        elif line.startswith("+"):
            body = line[1:]
            added.append(body)
            if not body.strip():
                continue  # blank lines carry no location
            indent = len(body) - len(body.lstrip())
            if sig:
                pending += _brackets(body)
                if pending <= 0 and (sig == "decorator" or body.split("#")[0].rstrip().endswith(":")):
                    sig = None
                continue
            if block is not None and indent > block:
                continue  # body of a newly added def/class
            stripped = body.lstrip()
            if stripped.startswith("@"):
                block, pending = indent, _brackets(body)
                sig = "decorator" if pending > 0 else None
                continue
            m_def = DEF_RE.match(line)
            if m_def and m_def.group(1) not in removed_names:
                new_defs.append((max(old - 1, 1), indent))
                block, pending = indent, _brackets(body)
                done = pending <= 0 and body.split("#")[0].rstrip().endswith(":")
                sig = None if done else "def"
                continue
            block = None
            touched.add(max(old - 1, 1))  # an insertion sits between old-1 and old
        elif line.startswith("\\"):
            continue
        else:
            old += 1
            block, pending, sig = None, 0, None
    return touched, added, new_defs


def touched_old_lines(patch: str) -> tuple[set[int], list[str]]:
    """Backward-compatible view of diff_points (touched lines, added lines)."""
    touched, added, _ = diff_points(patch)
    return touched, added


def credit_new_defs(new_defs: list[tuple[int, int]], sc: list[tuple[int, int, str, str]],
                    src_lines: list[str]) -> set[str]:
    """Credit a new def to the innermost existing scope it is nested inside.

    A nested helper added inside method M credits M; a new method added to class
    C credits C; a new sibling function or method credits nothing.
    """
    out = set()
    for anchor, indent in new_defs:
        containing = sorted((s for s in sc if s[0] <= anchor <= s[1]), key=lambda s: s[1] - s[0])
        for start, end, name, kind in containing:
            line = src_lines[start - 1] if start - 1 < len(src_lines) else ""
            if indent > len(line) - len(line.lstrip()):
                out.add(name)
                break
    return out


def scopes(source: str) -> list[tuple[int, int, str, str]]:
    """(start, end, qualname, kind) for top-level defs, classes and methods
    (also those inside module-level if/try/with blocks; see pyast.py)."""
    tree = ast.parse(source)
    out = []
    for node in top_level(tree):
        if isinstance(node, ast.ClassDef):
            out.append((*span(node), node.name, "class"))
            for sub in methods(node):
                out.append((*span(sub), f"{node.name}.{sub.name}", "method"))
        else:
            out.append((*span(node), node.name, "function"))
    return out


def map_lines(lines: set[int], sc: list[tuple[int, int, str, str]]) -> tuple[list[str], bool]:
    """Innermost scope per line; returns (qualnames, touched_module_level)."""
    found: set[str] = set()
    module_level = False
    for ln in lines:
        best = None
        for start, end, name, kind in sc:
            if start <= ln <= end and (best is None or (end - start) < (best[1] - best[0])):
                best = (start, end, name, kind)
        if best is None:
            module_level = True
        else:
            found.add(best[2])
    return sorted(found), module_level


def landed_diff(rec: dict) -> tuple[str, list[dict], str]:
    repo = rec["repo"]
    commit = gh.rest(f"repos/{repo}/commits/{rec['merge_commit']}")
    base = commit["parents"][0]["sha"]
    files = commit.get("files") or []
    paths = {f["filename"] for f in files}
    # Prefer the landed diff whenever it touches the PR's code: the PR file list can
    # still contain files of a stacked PR that landed separately (audit, AUDIT.md).
    if set(rec["core_files"]) & paths or rec.get("source_kind") == "commit":
        return base, files, "merge_commit_vs_first_parent"
    pr = gh.rest(f"repos/{repo}/pulls/{rec['pr_number']}")
    cmp_ = gh.rest(f"repos/{repo}/compare/{pr['base']['sha']}...{pr['head']['sha']}")
    return cmp_["merge_base_commit"]["sha"], cmp_.get("files") or [], "pr_diff_at_merge_base"


def build_gold(rec: dict) -> dict:
    repo = rec["repo"]
    base, files, how = landed_diff(rec)
    gold_files, gold_funcs, added_syms, added_lines = [], [], [], []
    module_level_files, parse_errors = [], []
    new_lines = sum(f["additions"] + f["deletions"] for f in files
                    if is_core_py(f["filename"]) and f.get("status") == "added")
    mod_lines = sum(f["additions"] + f["deletions"] for f in files
                    if is_core_py(f["filename"]) and f.get("status") != "added")
    for f in files:
        path = f["filename"]
        if not is_core_py(path) or f.get("status") == "added":
            continue
        patch = f.get("patch")
        if patch is None:  # binary or too large for the API
            parse_errors.append(path)
            continue
        old_path = f.get("previous_filename") or path
        gold_files.append(old_path)
        lines, added, new_defs = diff_points(patch)
        added_lines += added
        # only defs that are new in this diff (a re-wrapped `def` line also has a '-' twin)
        removed_defs = {m.group(1) for a in patch.splitlines()
                        if a.startswith("-") and (m := DEF_RE.match("+" + a[1:]))}
        added_syms += [m.group(1) for a in patch.splitlines()
                       if (m := DEF_RE.match(a)) and m.group(1) not in removed_defs]
        try:
            src = gh.rest(f"repos/{repo}/contents/{quote(old_path)}?ref={base}", raw=True)
            sc = scopes(src)
            qn, mod = map_lines(lines, sc)
            qn = sorted(set(qn) | credit_new_defs(new_defs, sc, src.splitlines()))
        except (SyntaxError, LookupError, ValueError) as e:
            parse_errors.append(f"{old_path}: {type(e).__name__}")
            continue
        gold_funcs += [f"{old_path}::{q}" for q in qn]
        if mod:
            module_level_files.append(old_path)

    text = f"{rec['issue_title']}\n{rec['issue_body']}"
    low = text.lower()
    func_names = {g.split("::")[1].split(".")[-1] for g in gold_funcs}
    substantive = {a.strip() for a in added_lines if len(a.strip()) >= 12}
    quoted = [a for a in substantive if a in text]
    rec = dict(rec)
    rec.pop("issue_body", None)  # keep the derived file free of issue text
    rec.update(
        base_commit=base,
        diff_source=how,
        gold_files=sorted(set(gold_files)),
        gold_functions=sorted(set(gold_funcs)),
        added_symbols=sorted(set(added_syms)),
        module_level_change_files=sorted(set(module_level_files)),
        parse_errors=parse_errors,
        lines_in_new_core_files=new_lines,
        lines_in_modified_core_files=mod_lines,
        flags={
            # most of the fix is brand-new code: modified files are often incidental
            "mostly_new_code": new_lines > mod_lines,
            # very old issues closed by a late PR are often only tangentially fixed (audit)
            "issue_older_than_1y": rec.get("issue_to_fix_hours", 0) > 8760,
            "issue_mentions_gold_file": any(
                g.lower() in low or g.rsplit("/", 1)[-1].lower() in low for g in gold_files
            ),
            "issue_mentions_gold_function": any(
                re.search(rf"\b{re.escape(n)}\b", text) for n in func_names if len(n) > 3
            ),
            "issue_quotes_added_code": len(quoted) >= 2 and len(quoted) >= 0.5 * max(len(substantive), 1),
        },
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+", help="*.candidates.jsonl files")
    ap.add_argument("--out", default="data/v0.1/gold.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="random sample per file (0 = all)")
    ap.add_argument("--cap-old-tiers", type=int, default=0,
                    help="per file, keep at most N T0 and N T1 records (T2 always kept)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(x)["instance_id"] for x in out.read_text(encoding="utf-8").split(chr(10)) if x.strip()}
    todo = []
    for name in args.candidates:
        src = Path(name)
        # split on newline only: str.splitlines() also splits on U+2028 inside issue text
        rows = [json.loads(x) for x in src.read_text(encoding="utf-8").split(chr(10)) if x.strip()]
        rng = random.Random(f"{args.seed}:{src.name}")
        if args.cap_old_tiers:
            keep = [r for r in rows if r["tier"] == "T2" or r.get("born_after_cutoff")]
            for t in ("T0", "T1"):
                pool = [r for r in rows if r["tier"] == t and not r.get("born_after_cutoff")]
                keep += rng.sample(pool, min(len(pool), args.cap_old_tiers))
            rows = keep
        if args.limit and len(rows) > args.limit:
            rows = rng.sample(rows, args.limit)
        todo += [r for r in rows if r["instance_id"] not in done]
    from concurrent.futures import ThreadPoolExecutor, as_completed

    uniq = list({r["instance_id"]: r for r in todo}.values())
    ok = 0
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_gold, rec): rec["instance_id"] for rec in uniq}
        for fut in as_completed(futs):
            try:
                g = fut.result()
            except Exception as e:  # keep going; record the failure
                print(f"  {futs[fut]}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)
                continue
            ok += 1
            fh.write(json.dumps(g, ensure_ascii=False) + chr(10))
            fh.flush()
            if ok % 100 == 0:
                print(f"  {ok}/{len(uniq)} gold records", flush=True)
    print(f"{ok} new gold records ({len(done)} already present) -> {out}", flush=True)


if __name__ == "__main__":
    main()
