"""Path classification shared by the harvester and the localization harness."""
from __future__ import annotations

import re

TEST_RE = re.compile(
    r"(^|/)(tests?|testing|test_utils)/|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$"
)
# Directories whose .py files are not "core" code (never gold targets).
NON_CORE_RE = re.compile(
    r"(^|/)(docs?|examples?|cookbooks?|notebooks?|samples?|contributing|"
    r"benchmarks?|scripts|\.github|\.ci|tutorials?|demo|demos)/"
)
BOT_LOGINS = {
    "dependabot", "renovate", "pre-commit-ci", "github-actions", "copilot",
    "copilot-swe-agent", "mergify", "allcontributors", "release-please",
}


def is_test(path: str) -> bool:
    return bool(TEST_RE.search(path))


def is_core_py(path: str) -> bool:
    return path.endswith(".py") and not is_test(path) and not NON_CORE_RE.search(path)


def is_bot(author: dict | None) -> bool:
    if not author:
        return True  # deleted ("ghost") accounts: treat as unusable
    login = (author.get("login") or "").lower()
    if author.get("__typename") == "Bot" or login.endswith("[bot]"):
        return True
    return login in BOT_LOGINS
