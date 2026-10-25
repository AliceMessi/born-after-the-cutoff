"""Repository snapshots at a base commit: download, list core files, skeletons.

Snapshots are GitHub tarballs extracted with tarfile's 'data' filter into
data/repos/<owner>__<repo>@<sha12>/. The code is only read as text; nothing
from a snapshot is ever imported or executed.
"""
from __future__ import annotations

import ast
import io
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harvest"))
from paths import is_core_py  # noqa: E402
from pyast import methods, top_level  # noqa: E402

REPOS = Path(__file__).resolve().parent.parent / "data" / "repos"


def snapshot(repo: str, sha: str) -> Path:
    dest = REPOS / f"{repo.replace('/', '__')}@{sha[:12]}"
    if dest.exists():
        return dest
    raw = subprocess.run(
        ["gh", "api", f"repos/{repo}/tarball/{sha}"], capture_output=True, check=True
    ).stdout
    tmp = dest.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".py")]
        tf.extractall(tmp, members=members, filter="data")
    # GitHub wraps everything in one top-level folder: strip it
    (top,) = list(tmp.iterdir())
    top.rename(dest)
    tmp.rmdir()
    return dest


def core_files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if is_core_py(p.relative_to(root).as_posix())
    )


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")


def functions(root: Path, rel: str) -> list[tuple[str, str]]:
    """(qualname, source) for top-level functions, classes and methods."""
    src = read(root, rel)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    out = []

    def seg(n):
        return "\n".join(lines[n.lineno - 1 : n.end_lineno])

    for node in top_level(tree):
        out.append((node.name, seg(node)))
        if isinstance(node, ast.ClassDef):
            out += [(f"{node.name}.{sub.name}", seg(sub)) for sub in methods(node)]
    return out


def skeleton(root: Path, rel: str, max_lines: int = 120) -> str:
    """Agentless-style compressed view: class/def signatures only."""
    src = read(root, rel)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "(unparseable)"
    lines = src.splitlines()
    out = []
    for node in top_level(tree):
        out.append(lines[node.lineno - 1].rstrip())
        if isinstance(node, ast.ClassDef):
            out += ["    " + lines[sub.lineno - 1].strip() for sub in methods(node)]
    return "\n".join(out[:max_lines])
