"""Map model-written locations onto canonical snapshot ids (prediction time).

Files:     exact repository-relative path, else a unique path-suffix match, else
           (pilot finding) a unique match on the guess's trailing components with at
           least `dir/file.py` left, so a wrong root folder (`src/pkg/x.py` for
           `pkg_slim/pkg/x.py`) does not erase an otherwise correct module.
Functions: canonical `path::Qual.name` among the snapshot's AST units. A bare name
           maps to `Class.name` only when it is unique in that file; dotted module
           ids (`pkg.mod.Class.method`) are accepted; junk suffixes such as `.def` or
           `()` are stripped. With `bare_ok`, a qualname without any path is accepted
           when it is unique among the candidate files (used for A step 2 and G1,
           whose prompts show the candidate files).
Unmappable predictions are dropped (they can never match gold).
"""
from __future__ import annotations

import re

PATH_RE = re.compile(r"[\w./-]+\.py\b")
FUNC_RE = re.compile(r"([\w./-]+\.py)\s*::\s*([\w.]+)")
QUAL_RE = re.compile(r"(?<![\w./])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\(\))?")
JUNK = (".def", ".py", ".method", ".function")


def norm_file(p: str, files: list[str]) -> str | None:
    p = p.strip().strip("`'\"").lstrip("./").replace("\\", "/")
    if p in files:
        return p
    hits = [f for f in files if f.endswith("/" + p)]
    if len(hits) == 1:
        return hits[0]
    parts = p.split("/")
    for k in range(1, len(parts) - 1):  # drop leading components, keep >= dir/file.py
        tail = "/".join(parts[k:])
        hits = [f for f in files if f == tail or f.endswith("/" + tail)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            break
    return None


def files_from_text(text: str, files: list[str]) -> list[str]:
    out = []
    for p in PATH_RE.findall(text):
        f = norm_file(p, files)
        if f and f not in out:
            out.append(f)
    return out


def _clean_qual(q: str) -> str:
    q = q.strip(".").removesuffix("()")
    for j in JUNK:
        if q.endswith(j):
            q = q[: -len(j)]
    return q


def norm_func(path: str, qual: str, units_by_file: dict[str, list[str]], files: list[str]) -> str | None:
    f = norm_file(path, files)
    qual = _clean_qual(qual)
    if not f:
        return None
    units = units_by_file.get(f, [])
    if qual in units:
        return f"{f}::{qual}"
    tail = qual.split(".")[-1]
    hits = [u for u in units if u == tail or u.endswith("." + tail)]
    if len(hits) == 1:
        return f"{f}::{hits[0]}"
    for u in units:  # "Class.method" written with an extra prefix (e.g. module path)
        if qual.endswith("." + u):
            return f"{f}::{u}"
    return None


def _bare(qual: str, units_by_file: dict[str, list[str]]) -> str | None:
    qual = _clean_qual(qual)
    if not qual:
        return None
    hits = [f"{f}::{u}" for f, us in units_by_file.items() for u in us
            if u == qual or u.endswith("." + qual) or qual.endswith("." + u)]
    hits = list(dict.fromkeys(hits))
    return hits[0] if len(hits) == 1 else None


def funcs_from_text(text: str, units_by_file: dict[str, list[str]], files: list[str],
                    bare_ok: bool = False) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        m = FUNC_RE.search(line)
        key = None
        if m:
            key = norm_func(m.group(1), m.group(2), units_by_file, files)
            if key is None and bare_ok:  # e.g. a copied placeholder path with a real qualname
                key = _bare(m.group(2), units_by_file)
        elif bare_ok:
            for q in QUAL_RE.findall(line.split("|")[0]):
                if "." in q or "_" in q or any(c.isupper() for c in q):
                    key = _bare(q, units_by_file)
                    if key:
                        break
        if key and key not in out:
            out.append(key)
    return out
