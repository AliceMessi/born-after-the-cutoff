"""Training-free retrieval baselines over a snapshot.

R0-random     seeded random file order (floor)
R0-path       BM25(issue, file path tokens): guessability control for the probe
B1-bm25       BM25(issue, path + file content) for files;
              BM25(issue, path::qualname + unit source) for functions

Tokenisation is cached per git blob id, so consecutive snapshots of the
same repository re-tokenise only the files that changed.
"""
from __future__ import annotations

import ast
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # third-party sources with odd escapes
import random
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

from bm25 import BM25, tokenize

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harvest"))
from paths import is_core_py  # noqa: E402
from pyast import methods, span, top_level  # noqa: E402


class LRU(OrderedDict):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def get_or(self, key, make):
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = make()
        self[key] = val
        if len(self) > self.cap:
            self.popitem(last=False)
        return val


FILE_TF = LRU(6000)
UNIT_TF = LRU(6000)


def blob_ids(root: Path) -> dict[str, str]:
    """path -> blob sha for the checked-out HEAD (core .py files only)."""
    p = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "HEAD"], cwd=root,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    out = {}
    for line in p.stdout.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob" and is_core_py(path):
            out[path] = parts[2]
    return out


def _units(src: str) -> list[tuple[str, str]]:
    """(qualname, source) units: functions, class headers and methods (pyast rules)."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    lines = src.splitlines()
    out = []

    nl = chr(10)

    def seg(n):
        a, b = span(n)
        return nl.join(lines[a - 1 : b])

    for n in top_level(tree):
        if isinstance(n, ast.ClassDef):
            body_defs = list(methods(n))
            head_end = min(span(d)[0] for d in body_defs) - 1 if body_defs else n.end_lineno
            out.append((n.name, nl.join(lines[span(n)[0] - 1 : head_end])))
            out += [(f"{n.name}.{d.name}", seg(d)) for d in body_defs]
        else:
            out.append((n.name, seg(n)))
    return out


class Snapshot:
    def __init__(self, root: Path):
        self.root = root
        self.blobs = blob_ids(root)
        self.files = sorted(self.blobs)

    def text(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8", errors="replace")

    def file_tf(self, path: str) -> Counter:
        return FILE_TF.get_or(self.blobs[path], lambda: Counter(tokenize(self.text(path))))

    def unit_tfs(self, path: str) -> list[tuple[str, Counter]]:
        return UNIT_TF.get_or(self.blobs[path], lambda: [(q, Counter(tokenize(q + "\n" + s)))
                                                         for q, s in _units(self.text(path))])


def rank_random(snap: Snapshot, seed: str) -> list[str]:
    files = list(snap.files)
    random.Random(seed).shuffle(files)
    return files


def rank_path_bm25(snap: Snapshot, query: str) -> list[str]:
    bm = BM25(tf=[Counter(tokenize(f)) for f in snap.files])
    return [snap.files[i] for i in bm.rank(query)]


def rank_bm25_files(snap: Snapshot, query: str) -> list[str]:
    tfs = [snap.file_tf(f) + Counter(tokenize(f)) for f in snap.files]
    bm = BM25(tf=tfs)
    return [snap.files[i] for i in bm.rank(query)]


def rank_bm25_functions(snap: Snapshot, query: str, top: int = 50) -> list[str]:
    keys, tfs = [], []
    path_tf = {}
    for f in snap.files:
        ptf = path_tf.setdefault(f, Counter(tokenize(f)))
        for q, tf in snap.unit_tfs(f):
            keys.append(f"{f}::{q}")
            tfs.append(tf + ptf)
    if not keys:
        return []
    bm = BM25(tf=tfs)
    return list(dict.fromkeys(keys[i] for i in bm.rank(query)))[:top]
