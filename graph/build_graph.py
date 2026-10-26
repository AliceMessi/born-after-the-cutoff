"""Build a repository code graph in the Gemma 4 Developer Agent competition's
public schema (NetworkX node-link JSON, directed multigraph).

Schema as published on the competition data page (visible without joining):
  directed, multigraph, graph{...}, nodes[{id, name, text}], edges[{source, target, type, key}]
Node ids are fully qualified dotted symbols (module, class, function/method),
e.g. `fastapi.routing._prepare_response_content`. Extra node fields we add
(`type`, `file`, `start_line`, `end_line`) are ignored by NetworkX consumers.
Compatibility is claimed only for these publicly documented fields.

Edge types: contains (module->class/function, class->method), imports
(module->module/symbol), calls (function/method->function/method/class),
inherits (class->class). Only statically resolvable targets inside the
repository become edges; unresolved calls are counted in graph.stats.

Two stages: `extract()` parses one file into a plain record (cacheable by git
blob id, so consecutive snapshots only re-parse changed files), and `build()`
resolves names across the snapshot. Source is only parsed with `ast`; nothing
is imported or executed.
"""
from __future__ import annotations

import ast
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # third-party sources with odd escapes
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harvest"))
from pyast import _flatten  # noqa: E402

BUILDER = "agentloc-graph/0.3"
MODULE_TEXT_LIMIT = 20000
_BUILTINS = set(dir(__import__("builtins")))
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def module_id(path: str, all_files: set[str]) -> str:
    """Dotted module id for a repository-relative .py path.

    Rule 1: if the path has a `src` directory, the module path starts after the last one.
    Rule 2: otherwise climb the contiguous chain of directories that contain __init__.py.
    """
    parts = path[:-3].split("/")
    is_init = parts[-1] == "__init__"
    dirs = path.split("/")[:-1]
    if "src" in dirs:
        i = len(dirs) - 1 - dirs[::-1].index("src")
        mod = parts[i + 1 :]
    else:
        k = len(dirs)
        while k > 0 and "/".join(dirs[:k]) + "/__init__.py" in all_files:
            k -= 1
        mod = parts[k:]
    if is_init:
        mod = mod[:-1]
    return ".".join(mod) or parts[-1]


def _dotted(expr: ast.AST) -> str | None:
    """'a.b.c' for Name/Attribute chains; 'super()' prefix for super().x."""
    out = []
    while isinstance(expr, ast.Attribute):
        out.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        out.append(expr.id)
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "super":
        out.append("super()")
    else:
        return None
    return ".".join(reversed(out))


def _span(n: ast.AST) -> tuple[int, int]:
    return min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]), n.end_lineno


def _scan(node: ast.AST, imports: list, calls: list | None, counts: dict) -> None:
    """One iterative walk collecting imports and (optionally) call targets."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Import):
            imports.extend(("import", a.name, a.asname, None, 0) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            imports.extend(("from", a.name, a.asname, n.module or "", n.level or 0) for a in n.names)
        elif calls is not None and isinstance(n, ast.Call):
            d = _dotted(n.func)
            if d:
                calls.append(d)
            else:
                counts["calls_dynamic"] = counts.get("calls_dynamic", 0) + 1
        stack.extend(ast.iter_child_nodes(n))


def extract(path: str, src: str) -> dict | None:
    """Parse one file into a plain, cacheable record (None if unparseable)."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    lines = src.splitlines()

    def text(n):
        s, e = _span(n)
        return "\n".join(lines[s - 1 : e])

    rec = {"path": path, "text": src[:MODULE_TEXT_LIMIT], "n_lines": len(lines), "defs": [], "classes": [],
           "imports": [], "counts": {}}
    # pyast._flatten looks through module-level if/try/with blocks (conditional definitions)
    for n in _flatten(tree.body):
        if isinstance(n, _DEFS):
            calls: list[str] = []
            _scan(n, rec["imports"], calls, rec["counts"])
            rec["defs"].append({"name": n.name, "kind": "function", "span": _span(n), "text": text(n),
                                "calls": calls, "cls": None})
        elif isinstance(n, ast.ClassDef):
            rec["classes"].append({"name": n.name, "span": _span(n), "text": text(n),
                                   "bases": [d for d in (_dotted(b) for b in n.bases) if d]})
            for s in _flatten(n.body):
                if isinstance(s, _DEFS):
                    calls = []
                    _scan(s, rec["imports"], calls, rec["counts"])
                    rec["defs"].append({"name": f"{n.name}.{s.name}", "kind": "method", "span": _span(s),
                                        "text": text(s), "calls": calls, "cls": n.name})
                else:
                    _scan(s, rec["imports"], None, rec["counts"])
        else:
            _scan(n, rec["imports"], None, rec["counts"])
    return rec


def build(root: Path, files: list[str], repo: str = "", commit: str = "",
          cache: dict | None = None, blob_of: dict[str, str] | None = None) -> dict:
    """Graph dict for the given repository-relative .py files under root.

    cache/blob_of: optional {blob_id: extract-record} cache and {path: blob_id}
    map; records are reused across snapshots when the blob id is unchanged.
    """
    fileset = set(files)
    nodes: dict[str, dict] = {}
    edges: set[tuple[str, str, str]] = set()
    stats: dict[str, int] = defaultdict(int)
    mods = []  # (module id, is_pkg, record)

    def add(nid, kind, path, text, span):
        if nid in nodes:
            stats["duplicate_ids"] += 1
            return
        nodes[nid] = {"id": nid, "name": nid, "text": text, "type": kind, "file": path,
                      "start_line": span[0], "end_line": span[1]}

    for path in sorted(files):
        key = blob_of.get(path) if blob_of else None
        rec = cache.get(key) if (cache is not None and key) else None
        if rec is None:
            try:
                src = (root / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                src = None
            rec = extract(path, src) if src is not None else None
            if cache is not None and key:
                cache[key] = rec if rec is not None else False
        if not rec:
            stats["unparseable_files"] += 1
            continue
        if rec["path"] != path:  # same blob at a different path: re-label
            rec = dict(rec, path=path)
        mid = module_id(path, fileset)
        if mid in nodes:  # e.g. two tests/conftest.py outside packages: use the full path
            mid = path[:-3].replace("/", ".")
            stats["module_id_collisions"] += 1
        mods.append((mid, path.endswith("__init__.py"), rec))
        add(mid, "module", path, rec["text"], (1, rec["n_lines"]))
        for c in rec["classes"]:
            add(f"{mid}.{c['name']}", "class", path, c["text"], c["span"])
            edges.add((mid, f"{mid}.{c['name']}", "contains"))
        for d in rec["defs"]:
            nid = f"{mid}.{d['name']}"
            add(nid, d["kind"], path, d["text"], d["span"])
            parent = f"{mid}.{d['cls']}" if d["cls"] else mid
            edges.add((parent, nid, "contains"))
        for k, v in rec["counts"].items():
            stats[k] += v

    def package(mid: str, is_pkg: bool, level: int) -> str:
        parts = mid.split(".")
        base = parts if is_pkg else parts[:-1]
        return ".".join(base[: len(base) - (level - 1)] if level > 1 else base)

    aliases_of = {}
    for mid, is_pkg, rec in mods:
        aliases: dict[str, str] = {}
        for kind, name, asname, module, level in rec["imports"]:
            if kind == "import":
                if asname:
                    aliases[asname] = name
                else:
                    top = name.split(".")[0]
                    aliases.setdefault(top, top)
                if name in nodes:
                    edges.add((mid, name, "imports"))
            else:
                if name == "*":
                    continue
                base = module
                if level:
                    pkg = package(mid, is_pkg, level)
                    base = f"{pkg}.{base}" if base and pkg else (base or pkg)
                target = f"{base}.{name}" if base else name
                aliases[asname or name] = target
                if target in nodes:
                    edges.add((mid, target, "imports"))
                elif base in nodes:
                    edges.add((mid, base, "imports"))
        aliases_of[mid] = aliases

    def resolve(mid: str, dotted: str, cls: str | None, bases: list[str]) -> str | None:
        head, _, rest = dotted.partition(".")
        cands = []
        if head in ("self", "cls") and cls and rest:
            cands.append(f"{cls}.{rest}")
            cands += [f"{b}.{rest}" for b in bases]
        elif head == "super()" and rest:
            cands += [f"{b}.{rest}" for b in bases]
        else:
            cands.append(f"{mid}.{dotted}")
            al = aliases_of[mid]
            if head in al:
                cands.append(al[head] + (f".{rest}" if rest else ""))
        for c in cands:
            if c in nodes:
                return c
        return None

    class_bases: dict[str, list[str]] = {}
    for mid, _, rec in mods:
        for c in rec["classes"]:
            cid = f"{mid}.{c['name']}"
            bs = []
            for d in c["bases"]:
                r = resolve(mid, d, None, [])
                if r and nodes[r]["type"] == "class":
                    bs.append(r)
                    edges.add((cid, r, "inherits"))
            class_bases[cid] = bs

    for mid, _, rec in mods:
        for d in rec["defs"]:
            uid = f"{mid}.{d['name']}"
            cid = f"{mid}.{d['cls']}" if d["cls"] else None
            bases = class_bases.get(cid, []) if cid else []
            for call in d["calls"]:
                r = resolve(mid, call, cid, bases)
                if r and r != uid:
                    edges.add((uid, r, "calls"))
                    stats["calls_resolved"] += 1
                elif not r:
                    stats["calls_builtin" if call.split(".")[0] in _BUILTINS else "calls_unresolved"] += 1

    keyed, counter = [], defaultdict(int)
    for s, t, kind in sorted(edges):
        keyed.append({"source": s, "target": t, "type": kind, "key": counter[(s, t)]})
        counter[(s, t)] += 1
    stats["nodes"] = len(nodes)
    stats["edges"] = len(keyed)
    return {
        "directed": True,
        "multigraph": True,
        "graph": {"repo": repo, "commit": commit, "builder": BUILDER, "stats": dict(stats)},
        "nodes": list(nodes.values()),
        "edges": keyed,
    }


def gold_to_ids(gold_functions: list[str], graph: dict) -> list[str | None]:
    """Map gold `path::Class.method` entries to dotted graph ids (None if absent)."""
    mod_of_file = {n["file"]: n["id"] for n in graph["nodes"] if n["type"] == "module"}
    ids = {n["id"] for n in graph["nodes"]}
    out = []
    for g in gold_functions:
        path, _, qual = g.partition("::")
        mod = mod_of_file.get(path)
        cand = f"{mod}.{qual}" if mod else None
        out.append(cand if cand in ids else None)
    return out


def save(graph: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(graph, fh)
