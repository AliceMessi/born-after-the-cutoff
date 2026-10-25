"""Map model-written locations onto canonical snapshot ids (prediction time).

Files:     exact repository-relative path, else a unique path-suffix match.
Functions: canonical `path::Qual.name` among the snapshot's AST units. A bare
           name maps to `Class.name` only when it is unique in that file;
           dotted module ids (`pkg.mod.Class.method`) are also accepted.
Unmappable predictions are dropped (they can never match gold).
"""
from __future__ import annotations

import re

PATH_RE = re.compile(r"[\w./-]+\.py\b")
FUNC_RE = re.compile(r"([\w./-]+\.py)\s*::\s*([\w.]+)")


def norm_file(p: str, files: list[str]) -> str | None:
    p = p.strip().lstrip("./").replace("\\", "/")
    if p in files:
        return p
    hits = [f for f in files if f.endswith("/" + p)]
    return hits[0] if len(hits) == 1 else None


def files_from_text(text: str, files: list[str]) -> list[str]:
    out = []
    for p in PATH_RE.findall(text):
        f = norm_file(p, files)
        if f and f not in out:
            out.append(f)
    return out


def norm_func(path: str, qual: str, units_by_file: dict[str, list[str]], files: list[str]) -> str | None:
    f = norm_file(path, files)
    if not f:
        return None
    units = units_by_file.get(f, [])
    qual = qual.strip(".")
    if qual in units:
        return f"{f}::{qual}"
    tail = qual.split(".")[-1]
    hits = [u for u in units if u == tail or u.endswith("." + tail)]
    if len(hits) == 1:
        return f"{f}::{hits[0]}"
    # "Class.method" written with an extra prefix (e.g. module path)
    for u in units:
        if qual.endswith("." + u):
            return f"{f}::{u}"
    return None


def funcs_from_text(text: str, units_by_file: dict[str, list[str]], files: list[str]) -> list[str]:
    out = []
    for path, qual in FUNC_RE.findall(text):
        k = norm_func(path, qual, units_by_file, files)
        if k and k not in out:
            out.append(k)
    return out
