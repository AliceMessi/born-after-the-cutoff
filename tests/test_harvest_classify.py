from harvest import CLOSE_RE, classify, linked_issue_numbers, tier_of

BODY = "When I call Runner.run with a tool that returns None the tracing exporter crashes. " * 2


def pr(files, title="Fix crash", issue_created="2026-05-01T10:00:00Z", pr_created="2026-05-03T10:00:00Z",
       issue_author="alice", pr_author="bob"):
    return {
        "kind": "pr", "number": 7, "title": title, "body": "Fixes #3", "createdAt": pr_created,
        "mergedAt": "2026-05-04T10:00:00Z", "baseRefName": "main", "mergeCommit": {"oid": "a" * 40},
        "commits": {"totalCount": 1}, "author": {"login": pr_author, "__typename": "User"},
        "files": {"totalCount": len(files), "nodes": files},
        "issues": [{"number": 3, "title": "Crash", "body": BODY, "createdAt": issue_created,
                    "author": {"login": issue_author, "__typename": "User"}}],
    }


def f(path, add=5, dele=2, kind="MODIFIED"):
    return {"path": path, "additions": add, "deletions": dele, "changeType": kind}


GOOD = [f("src/agents/run.py"), f("tests/test_run.py")]


def test_ok_and_fields():
    reason, rec = classify("o/r", pr(GOOD), "2024-01-01", "2025-03-11T00:00:00Z")
    assert reason == "ok"
    assert rec["tier"] == "T2" and rec["born_after_cutoff"] and not rec["same_author"]
    assert rec["issue_to_fix_hours"] == 48.0
    assert "alice" not in str(rec) and "bob" not in str(rec)  # logins are never stored


def test_filters():
    created = "2020-01-01T00:00:00Z"
    assert classify("o/r", pr([f("tests/test_x.py")]), "2024-01-01", created)[0] == "no_core_py_change"
    assert classify("o/r", pr([f("src/a.py")]), "2024-01-01", created)[0] == "no_test_change"
    big = [f(f"src/m{i}.py") for i in range(6)] + [f("tests/test_a.py")]
    assert classify("o/r", pr(big), "2024-01-01", created)[0] == "too_large"
    assert classify("o/r", pr([f("src/a.py", 1, 0), f("tests/t_test.py")]), "2024-01-01", created)[0] == "trivial"
    new_only = [f("src/new.py", kind="ADDED"), f("tests/test_new.py")]
    assert classify("o/r", pr(new_only), "2024-01-01", created)[0] == "only_new_files"
    late = pr(GOOD, issue_created="2026-05-05T00:00:00Z")
    assert classify("o/r", late, "2024-01-01", created)[0] == "issue_after_fix"
    old = pr(GOOD, issue_created="2023-05-01T00:00:00Z", pr_created="2023-06-01T00:00:00Z")
    assert classify("o/r", old, "2024-01-01", created)[0] == "issue_before_window"


def test_tiers():
    assert tier_of("2025-01-31T23:59:59Z") == "T0"
    assert tier_of("2025-02-01T00:00:00Z") == "T1"
    assert tier_of("2026-04-01T12:00:00Z") == "T1"
    assert tier_of("2026-04-02T00:00:00Z") == "T2"


def test_closing_keyword_fallback():
    p = {"closingIssuesReferences": {"nodes": []}, "body": "This fixes #12 and Closes: #15; refs #99"}
    assert linked_issue_numbers("o/r", p) == [12, 15]
    assert CLOSE_RE.findall("resolved #4") == ["4"]
