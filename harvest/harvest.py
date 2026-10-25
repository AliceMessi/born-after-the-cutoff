"""Harvest issue -> fix localization candidates from GitHub.

Three modes, all writing the same candidate schema:
  prs      (default) merged PRs per month window, linked to the issue they close
           (closingIssuesReferences, else closing keywords in the PR body)
  commits  issues whose ClosedEvent closer is a commit (repos synced by Copybara,
           e.g. google/adk-python, close issues with direct commits)
  curated  explicit PR -> issue pairs from a JSON file (our OpenMinion slice);
           every record is kept and the filter verdict is stored in `filter_reason`

Tiers are assigned from the issue creation date (T0 < 2025-02-01 <= T1 < 2026-04-02 <= T2),
so a single continuous window per repository loses nothing at tier boundaries.

Usage (from the project folder):
  python -I harvest/harvest.py --repo openai/openai-agents-python --since 2025-03-01 --out data/v0.1/raw
  python -I harvest/harvest.py --repo google/adk-python --mode commits --since 2025-04-01 --out data/v0.1/raw
  python -I harvest/harvest.py --repo openminion/openminion --mode curated \
      --pairs harvest/openminion_pairs.json --out data/v0.1/raw

Phase 3 (gold.py) maps the landed diff to files and functions at the base commit.
All GitHub content is treated as data; nothing is posted anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh  # noqa: E402
from paths import is_bot, is_core_py, is_test  # noqa: E402

PRETRAIN_CUTOFF = "2025-02-01"  # Gemma 4 model card: pre-training data cutoff January 2025
RELEASE_DATE = "2026-04-02"  # Gemma 4 public release (Kaggle Models instances created that day)
RELEASE_12B = "2026-06-04"  # Gemma 4 12B published 3 June 2026: strict tier for 12B

SEARCH_PRS = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number createdAt mergedAt baseRefName body
      author { login __typename }
      closingIssuesReferences(first: 3) { totalCount nodes { number repository { nameWithOwner } } }
    } }
  }
}"""

SEARCH_CLOSED_ISSUES = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 50, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on Issue {
      number title body createdAt closedAt state
      author { login __typename }
      timelineItems(itemTypes: [CLOSED_EVENT], last: 1) { nodes { ... on ClosedEvent {
        closer { __typename
          ... on Commit { oid authoredDate committedDate messageHeadline message
                          author { user { login } }
                          associatedPullRequests(first: 1) { nodes { number merged } } }
          ... on PullRequest { number }
        } } } }
    } }
  }
}"""

DETAIL_FIELDS = """
      number title body createdAt mergedAt baseRefName baseRefOid headRefOid
      mergeCommit { oid } additions deletions changedFiles
      commits { totalCount }
      author { login __typename }
      files(first: 100) { totalCount nodes { path additions deletions changeType } }
"""
ISSUE_FIELDS = """
      number title body createdAt state
      author { login __typename }
"""

# Closing keywords in a PR body (GitHub only auto-links them for PRs into the
# default branch, so repos that merge into `dev` need this fallback).
CLOSE_RE = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+#(\d+)", re.I)
TRIVIAL_TITLE_RE = re.compile(r"\b(typo|spelling|readme|docs?|bump|release|changelog|lint|format(ting)?)\b", re.I)
STATUS_TO_CHANGE = {"added": "ADDED", "removed": "DELETED", "modified": "MODIFIED", "renamed": "RENAMED"}


def month_windows(since: str, until: str):
    a = dt.date.fromisoformat(since)
    end = dt.date.fromisoformat(until)
    while a <= end:
        nxt = (a.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        b = min(nxt - dt.timedelta(days=1), end)
        yield a.isoformat(), b.isoformat()
        a = nxt


def tier_of(issue_created: str) -> str:
    d = issue_created[:10]
    if d < PRETRAIN_CUTOFF:
        return "T0"
    return "T1" if d < RELEASE_DATE else "T2"


def hours_between(a: str, b: str) -> float:
    fa = dt.datetime.fromisoformat(a.replace("Z", "+00:00"))
    fb = dt.datetime.fromisoformat(b.replace("Z", "+00:00"))
    return round((fb - fa).total_seconds() / 3600, 2)


# ---------------------------------------------------------------- PR mode
def search_prs(repo: str, since: str, until: str) -> list[dict]:
    prs = []
    for a, b in month_windows(since, until):
        cursor = None
        while True:
            q = f"repo:{repo} is:pr is:merged merged:{a}..{b}"
            out = gh.graphql(SEARCH_PRS, {"q": q, "cursor": cursor})["data"]["search"]
            if out["issueCount"] > 1000:
                print(f"  warning: {q} has {out['issueCount']} results (>1000 cap)", file=sys.stderr)
            prs.extend(n for n in out["nodes"] if n)
            if not out["pageInfo"]["hasNextPage"]:
                break
            cursor = out["pageInfo"]["endCursor"]
    return prs


def linked_issue_numbers(repo: str, pr: dict) -> list[int]:
    nums = [
        n["number"]
        for n in pr["closingIssuesReferences"]["nodes"]
        if n["repository"]["nameWithOwner"].lower() == repo.lower()
    ]
    if not nums:
        nums = [int(m) for m in CLOSE_RE.findall(pr.get("body") or "")]
    return sorted(set(nums))[:3]


def pr_details(repo: str, wanted: dict[int, list[int]], batch: int = 10) -> dict[int, dict]:
    owner, name = repo.split("/")
    details: dict[int, dict] = {}
    numbers = sorted(wanted)
    for i in range(0, len(numbers), batch):
        chunk = numbers[i : i + batch]
        parts = [f"p{n}: pullRequest(number: {n}) {{ {DETAIL_FIELDS} }}" for n in chunk]
        issue_nums = sorted({x for n in chunk for x in wanted[n]})
        parts += [f"i{x}: issue(number: {x}) {{ {ISSUE_FIELDS} }}" for x in issue_nums]
        q = f'query {{ repository(owner: "{owner}", name: "{name}") {{ {" ".join(parts)} }} }}'
        try:
            data = gh.graphql(q, {})["data"]["repository"]
        except (RuntimeError, LookupError) as e:  # e.g. a referenced number is a PR, not an issue
            print(f"  batch error, retrying one by one: {str(e)[:100]}", file=sys.stderr)
            data = {}
            for n in chunk:
                q1 = f'query {{ repository(owner: "{owner}", name: "{name}") {{ p{n}: pullRequest(number: {n}) {{ {DETAIL_FIELDS} }} }} }}'
                try:
                    data.update(gh.graphql(q1, {})["data"]["repository"])
                except (RuntimeError, LookupError):
                    data[f"p{n}"] = None
            for x in issue_nums:
                q1 = f'query {{ repository(owner: "{owner}", name: "{name}") {{ i{x}: issue(number: {x}) {{ {ISSUE_FIELDS} }} }} }}'
                try:
                    data.update(gh.graphql(q1, {})["data"]["repository"])
                except (RuntimeError, LookupError):
                    data[f"i{x}"] = None
        for n in chunk:
            pr = data.get(f"p{n}")
            if pr:
                pr["issues"] = [data.get(f"i{x}") for x in wanted[n] if data.get(f"i{x}")]
                pr["kind"] = "pr"
                details[n] = pr
    return details


# ------------------------------------------------------------ commit mode
def commit_closed(repo: str, since: str, until: str) -> list[dict]:
    """Issues closed by a direct commit, shaped like PR details for classify()."""
    out_items: dict[str, dict] = {}
    for a, b in month_windows(since, until):
        cursor = None
        while True:
            q = f"repo:{repo} is:issue is:closed closed:{a}..{b}"
            res = gh.graphql(SEARCH_CLOSED_ISSUES, {"q": q, "cursor": cursor})["data"]["search"]
            for iss in res["nodes"]:
                if not iss:
                    continue
                ev = (iss.get("timelineItems") or {}).get("nodes") or []
                closer = (ev[-1] or {}).get("closer") if ev else None
                if not closer or closer.get("__typename") != "Commit":
                    continue
                prs = (closer.get("associatedPullRequests") or {}).get("nodes") or []
                if any(p.get("merged") for p in prs):
                    continue  # PR flow: covered by PR mode
                oid = closer["oid"]
                item = out_items.get(oid)
                issue = {k: iss[k] for k in ("number", "title", "body", "createdAt", "state", "author")}
                if item:
                    item["issues"].append(issue)
                    continue
                out_items[oid] = {
                    "kind": "commit",
                    "number": None,
                    "title": closer["messageHeadline"],
                    "body": closer["message"],
                    "createdAt": closer["authoredDate"],
                    "mergedAt": closer["committedDate"],
                    "baseRefName": None,
                    "mergeCommit": {"oid": oid},
                    "commits": {"totalCount": 1},
                    "author": {"login": ((closer.get("author") or {}).get("user") or {}).get("login"),
                               "__typename": "User"},
                    "issues": [issue],
                }
            if not res["pageInfo"]["hasNextPage"]:
                break
            cursor = res["pageInfo"]["endCursor"]
    # file lists come from the REST commit endpoint
    for oid, item in out_items.items():
        try:
            c = gh.rest(f"repos/{repo}/commits/{oid}")
        except (RuntimeError, LookupError):
            item["files"] = {"totalCount": 0, "nodes": []}
            continue
        files = c.get("files") or []
        item["files"] = {
            "totalCount": len(files),
            "nodes": [{"path": f["filename"], "additions": f["additions"], "deletions": f["deletions"],
                       "changeType": STATUS_TO_CHANGE.get(f["status"], "MODIFIED")} for f in files],
        }
    return list(out_items.values())


# --------------------------------------------------------------- classify
def classify(repo: str, pr: dict, since: str, repo_created: str) -> tuple[str, dict]:
    """Return (reject_reason or 'ok', candidate record)."""
    issues = pr.get("issues") or []
    if not issues:
        return "no_issue_resolved", {}
    issue = min(issues, key=lambda x: x["createdAt"])  # earliest linked issue is the report
    files = pr["files"]["nodes"]
    core = [f for f in files if is_core_py(f["path"])]
    tests = [f for f in files if is_test(f["path"])]
    core_lines = sum(f["additions"] + f["deletions"] for f in core)
    text = f"{issue['title']}\n{issue.get('body') or ''}"
    oid = (pr.get("mergeCommit") or {}).get("oid")
    stem = repo.replace("/", "__")
    iid = f"{stem}-{pr['number']}" if pr["kind"] == "pr" else f"{stem}-c{oid[:10]}"
    rec = {
        "instance_id": iid,
        "repo": repo,
        "source_kind": pr["kind"],
        "pr_number": pr["number"],
        "issue_numbers": [x["number"] for x in issues],
        "issue_url": f"https://github.com/{repo}/issues/{issue['number']}",
        "fix_url": (f"https://github.com/{repo}/pull/{pr['number']}" if pr["kind"] == "pr"
                    else f"https://github.com/{repo}/commit/{oid}"),
        "issue_created_at": issue["createdAt"],
        "pr_created_at": pr["createdAt"],
        "merged_at": pr["mergedAt"],
        "issue_to_fix_hours": hours_between(issue["createdAt"], pr["createdAt"]),
        "base_ref": pr["baseRefName"],
        "merge_commit": oid,
        "pr_commits": pr["commits"]["totalCount"],
        "pr_author_bot": is_bot(pr.get("author")),
        # Logins are not stored (personal data); only whether reporter == fixer.
        "same_author": ((issue.get("author") or {}).get("login") or "?")
        == ((pr.get("author") or {}).get("login") or "!"),
        "issue_title": issue["title"],
        "issue_body": issue.get("body") or "",
        "core_files": [f["path"] for f in core],
        "core_added_files": [f["path"] for f in core if f["changeType"] == "ADDED"],
        "test_files": [f["path"] for f in tests],
        "core_lines_changed": core_lines,
        "files_truncated": pr["files"]["totalCount"] > len(files),
        "repo_created_at": repo_created,
        "born_after_cutoff": repo_created[:10] >= PRETRAIN_CUTOFF,
        "tier": tier_of(issue["createdAt"]),
        "after_12b_release": issue["createdAt"][:10] >= RELEASE_12B,
    }
    if issue["createdAt"][:10] < since:
        return "issue_before_window", rec
    if is_bot(issue.get("author")):
        return "issue_by_bot", rec
    if issue["createdAt"] > pr["createdAt"]:
        return "issue_after_fix", rec
    if not core:
        return "no_core_py_change", rec
    modified_core = [f for f in core if f["changeType"] != "ADDED"]
    if not modified_core:
        return "only_new_files", rec
    if len(core) > 5 or core_lines > 400 or rec["files_truncated"]:
        return "too_large", rec
    if core_lines < 3 or TRIVIAL_TITLE_RE.search(pr["title"] or "") and core_lines < 10:
        return "trivial", rec
    if not tests:
        return "no_test_change", rec
    if len(text.strip()) < 80:
        return "issue_too_short", rec
    if "```diff" in text or "diff --git" in text:
        return "issue_contains_diff", rec
    return "ok", rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mode", choices=["prs", "commits", "curated"], default="prs")
    ap.add_argument("--since", default=RELEASE_DATE)
    ap.add_argument("--until", default=dt.date.today().isoformat())
    ap.add_argument("--pairs", help="curated mode: JSON with {pairs: [{pr, issue}]}")
    ap.add_argument("--out", default="data/v0.1/raw")
    ap.add_argument("--tag", default="", help="suffix for output names (several windows per repo)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.repo.replace('/', '__')}.{args.mode}" + (f".{args.tag}" if args.tag else "")
    repo_created = gh.rest(f"repos/{args.repo}")["created_at"]

    funnel: Counter = Counter()
    if args.mode == "prs":
        prs = search_prs(args.repo, args.since, args.until)
        funnel["merged_prs"] = len(prs)
        wanted: dict[int, list[int]] = {}
        for pr in prs:
            if is_bot(pr.get("author")):
                funnel["reject:pr_by_bot"] += 1
                continue
            nums = linked_issue_numbers(args.repo, pr)
            if nums:
                wanted[pr["number"]] = nums
            else:
                funnel["reject:no_linked_issue"] += 1
        funnel["linked"] = len(wanted)
        items = list(pr_details(args.repo, wanted).values())
    elif args.mode == "commits":
        items = commit_closed(args.repo, args.since, args.until)
        funnel["linked"] = len(items)
    else:
        pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))["pairs"]
        wanted = defaultdict(list)
        for p in pairs:
            wanted[p["pr"]].append(p["issue"])
        items = list(pr_details(args.repo, dict(wanted)).values())
        funnel["linked"] = len(items)
        args.since = "2000-01-01"

    per_month: dict[str, Counter] = defaultdict(Counter)
    tiers: Counter = Counter()
    with (out / f"{stem}.candidates.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            reason, rec = classify(args.repo, it, args.since, repo_created)
            month = (it.get("mergedAt") or "?")[:7]
            per_month[month]["linked"] += 1
            if reason == "ok" or (args.mode == "curated" and rec):
                rec["filter_reason"] = reason
                if reason == "ok":
                    funnel["ok"] += 1
                    per_month[month]["ok"] += 1
                    tiers[rec["tier"]] += 1
                else:
                    funnel[f"kept_despite:{reason}"] += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                funnel[f"reject:{reason}"] += 1
    report = {
        "repo": args.repo, "mode": args.mode, "since": args.since, "until": args.until,
        "repo_created_at": repo_created, "funnel": dict(funnel), "ok_by_tier": dict(tiers),
        "per_month": {m: dict(c) for m, c in sorted(per_month.items())},
    }
    (out / f"{stem}.funnel.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.repo, args.mode, json.dumps({"funnel": report["funnel"], "tiers": report["ok_by_tier"]}), flush=True)


if __name__ == "__main__":
    main()
