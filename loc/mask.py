"""Identifier masking for the lexical-cue ablation (PREREG H5).

Replaces the code-like parts of an issue with placeholders while keeping the prose:
  * backticked spans                     -> [CODE]
  * file paths (*.py, a/b/c)             -> [PATH]
  * snake_case, CamelCase, dotted names  -> [SYMBOL]
  * other words that exactly match a symbol of the repository snapshot
    (function, method, class or module name, at least 4 characters) -> [SYMBOL]
Fenced code blocks are replaced by [CODE BLOCK]. Ordinary English words are kept
unless they are also repository symbols of length >= 4.
"""
from __future__ import annotations

import re

FENCE = re.compile(r"```.*?```", re.S)
TICK = re.compile(r"`[^`\n]+`")
PATH = re.compile(r"(?<![\w/])(?:[\w.-]+/)+[\w.-]+|\b[\w-]+\.py\b")
DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
WORD = re.compile(r"\b[A-Za-z_]\w*\b")


def _code_like(tok: str) -> bool:
    return "_" in tok or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok))


def mask_issue(text: str, symbols: set[str] | None = None) -> str:
    symbols = {s for s in (symbols or set()) if len(s) >= 4}
    t = FENCE.sub("[CODE BLOCK]", text or "")
    t = TICK.sub("[CODE]", t)
    t = PATH.sub(lambda m: "[PATH]" if ("/" in m.group(0) or m.group(0).endswith(".py")) else m.group(0), t)
    t = DOTTED.sub(lambda m: m.group(0) if m.group(0)[0].isdigit() else "[SYMBOL]", t)

    def word(m):
        w = m.group(0)
        if w.isupper() and len(w) <= 5:  # acronyms such as HTTP, JSON stay
            return w
        return "[SYMBOL]" if (_code_like(w) or w in symbols) else w

    return WORD.sub(word, t)


def snapshot_symbols(units_by_file: dict[str, list[str]], files: list[str]) -> set[str]:
    out = set()
    for f, quals in units_by_file.items():
        for q in quals:
            out.update(q.split("."))
    for f in files:
        out.add(f.rsplit("/", 1)[-1].removesuffix(".py"))
    return out
