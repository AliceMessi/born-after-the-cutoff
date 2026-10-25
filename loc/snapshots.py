"""Disk-frugal repository snapshots with git partial fetches.

One git store per repository under data/repos/git/<owner>__<repo>:
  * each base commit is fetched with `--depth=1 --filter=blob:none`;
  * the work tree is a sparse checkout of `*.py` plus licence files;
  * blobs are fetched lazily at checkout and deduplicated by git's object store,
    so hundreds of snapshots of the same repository cost little more than one.

The checked-out code is only read as text; it is never imported or executed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# AGENTLOC_REPO_STORE moves the stores (e.g. to /tmp on Kaggle so they are not saved as output)
STORE = Path(os.environ.get("AGENTLOC_REPO_STORE") or Path(__file__).resolve().parent.parent / "data" / "repos" / "git")
SPARSE = ["*.py", "LICENSE*", "LICENCE*", "COPYING*", "NOTICE*"]


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false",
         "-c", "advice.detachedHead=false", *args],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=check,
    )


def store_path(repo: str) -> Path:
    return STORE / repo.replace("/", "__")


def ensure_store(repo: str) -> Path:
    path = store_path(repo)
    if (path / ".git").exists():
        return path
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "remote", "add", "origin", f"https://github.com/{repo}.git")
    _git(path, "config", "core.sparseCheckout", "true")
    _git(path, "sparse-checkout", "set", "--no-cone", *SPARSE)
    return path


def has_commit(path: Path, sha: str) -> bool:
    return _git(path, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def checkout(repo: str, sha: str) -> Path:
    """Materialise the sparse work tree of `repo` at `sha` and return its path."""
    path = ensure_store(repo)
    if not has_commit(path, sha):
        _git(path, "fetch", "-q", "--depth=1", "--filter=blob:none", "origin", sha)
    head = _git(path, "rev-parse", "HEAD", check=False).stdout.strip()
    if head != sha:
        _git(path, "checkout", "-q", "--force", "--detach", sha)
    return path


def disk_usage(path: Path = STORE) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def prune(repo: str) -> None:
    """Drop a repository store entirely (re-fetchable on demand)."""
    shutil.rmtree(store_path(repo), ignore_errors=True)
