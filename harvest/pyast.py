"""Shared AST helpers so gold extraction, BM25 units and the code graph agree
on what a "function-level unit" is.

Top-level units are module-level functions and classes, including those defined
inside module-level `if` / `try` / `with` blocks (conditional imports and
fallback definitions such as `try: ... except ImportError: class X: ...`).
Methods are the defs directly inside a class body (also inside class-level
`if` / `try` blocks). Nested functions are part of their enclosing unit.
"""
from __future__ import annotations

import ast

DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_BLOCKS = (ast.If, ast.Try, ast.With, ast.AsyncWith) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())


def _flatten(stmts: list[ast.stmt]):
    for s in stmts:
        if isinstance(s, _BLOCKS):
            for field in ("body", "orelse", "finalbody"):
                yield from _flatten(getattr(s, field, []) or [])
            for h in getattr(s, "handlers", []) or []:
                yield from _flatten(h.body)
        else:
            yield s


def top_level(tree: ast.Module):
    """Module-level defs and classes, looking through if/try/with blocks."""
    for s in _flatten(tree.body):
        if isinstance(s, DEFS + (ast.ClassDef,)):
            yield s


def methods(cls: ast.ClassDef):
    for s in _flatten(cls.body):
        if isinstance(s, DEFS):
            yield s


def span(n: ast.AST) -> tuple[int, int]:
    """First line (including decorators) and last line of a def/class."""
    return min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]), n.end_lineno
