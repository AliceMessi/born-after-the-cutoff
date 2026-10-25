"""Load benchmark instances: gold labels joined with the (local-only) issue text."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V01 = ROOT / "data" / "v0.1"


def read_jsonl(path: Path) -> list[dict]:
    # split on newline only: str.splitlines() also splits on U+2028 inside issue text
    lines = [x for x in path.read_text(encoding="utf-8").split(chr(10)) if x.strip()]
    out = []
    for i, x in enumerate(lines):
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:
            if i != len(lines) - 1:  # only a half-written final line may be skipped
                raise
    return out


def is_independent(r: dict) -> bool:
    """Primary subset: reporter != fixer, >= 24 h before the fix, no quoted fix code."""
    return (not r["same_author"]) and r["issue_to_fix_hours"] >= 24 and not r["flags"]["issue_quotes_added_code"]


def load(gold_path: Path = V01 / "gold.jsonl", raw_dir: Path = V01 / "raw") -> list[dict]:
    texts = {}
    for f in raw_dir.glob("*.candidates.jsonl"):
        for r in read_jsonl(f):
            texts[r["instance_id"]] = (r["issue_title"], r["issue_body"])
    out = []
    for g in read_jsonl(gold_path):
        if not g["gold_files"]:
            continue  # nothing localisable at file level (e.g. only new files)
        title, body = texts.get(g["instance_id"], (g.get("issue_title", ""), ""))
        g["problem_statement"] = f"{title}\n\n{body}"
        g["independent"] = is_independent(g)
        out.append(g)
    return out
