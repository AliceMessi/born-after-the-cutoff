"""Build the v0.1 release files under data/v0.1/release/ (local; reviewed before any publication).

  instances.jsonl  labels and metadata only (no issue text): CC BY 4.0
  issues.jsonl     issue title + body for convenience, with @mentions and e-mail
                   addresses masked; each record keeps its source URL (attribution)
  stats.json       funnel and composition numbers for the datasheet

  python -I harvest/build_release.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loc"))
from data import V01, load, read_jsonl  # noqa: E402

MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LABEL_FIELDS = [
    "instance_id", "repo", "source_kind", "pr_number", "issue_numbers", "issue_url", "fix_url",
    "issue_created_at", "pr_created_at", "merged_at", "issue_to_fix_hours", "base_commit", "merge_commit",
    "diff_source", "tier", "after_12b_release", "born_after_cutoff", "repo_created_at", "same_author",
    "independent", "gold_files", "gold_functions", "added_symbols", "module_level_change_files",
    "core_lines_changed", "test_files", "flags", "filter_reason",
]


def mask(text: str) -> str:
    return MENTION.sub("@[user]", EMAIL.sub("[email]", text or ""))


def main() -> None:
    out = V01 / "release"
    out.mkdir(parents=True, exist_ok=True)
    insts = load()
    with (out / "instances.jsonl").open("w", encoding="utf-8") as fl, \
         (out / "issues.jsonl").open("w", encoding="utf-8") as fi:
        for r in insts:
            fl.write(json.dumps({k: r.get(k) for k in LABEL_FIELDS}, ensure_ascii=False) + chr(10))
            title, _, body = r["problem_statement"].partition("\n\n")
            fi.write(json.dumps({"instance_id": r["instance_id"], "source_url": r["issue_url"],
                                 "title": mask(title), "body": mask(body)}, ensure_ascii=False) + chr(10))
    funnels = [json.loads(p.read_text(encoding="utf-8")) for p in (V01 / "raw").glob("*.funnel.json")]
    stats = {
        "instances": len(insts),
        "by_tier": dict(Counter(r["tier"] for r in insts)),
        "born_after": sum(r["born_after_cutoff"] for r in insts),
        "independent": sum(r["independent"] for r in insts),
        "independent_T2": sum(r["independent"] and r["tier"] == "T2" for r in insts),
        "by_repo": dict(Counter(r["repo"] for r in insts).most_common()),
        "mentions_gold_file": sum(r["flags"]["issue_mentions_gold_file"] for r in insts),
        "mentions_gold_function": sum(r["flags"]["issue_mentions_gold_function"] for r in insts),
        "quotes_added_code": sum(r["flags"]["issue_quotes_added_code"] for r in insts),
        "same_author": sum(r["same_author"] for r in insts),
        "mean_gold_files": round(sum(len(r["gold_files"]) for r in insts) / max(len(insts), 1), 3),
        "mean_gold_functions": round(sum(len(r["gold_functions"]) for r in insts) / max(len(insts), 1), 3),
        "no_function_gold": sum(not r["gold_functions"] for r in insts),
        "funnels": {f"{f['repo']}|{f['mode']}|{f['since']}": f["funnel"] for f in funnels},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("funnels", "by_repo")}, indent=1))


if __name__ == "__main__":
    main()
