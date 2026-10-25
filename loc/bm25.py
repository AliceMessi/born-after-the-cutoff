"""Dependency-free Okapi BM25 (Robertson & Zaragoza, 2009) over code."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Split identifiers on snake_case and camelCase; lower-case."""
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25:
    def __init__(self, docs: list[str] | None = None, k1: float = 1.2, b: float = 0.75,
                 tf: list[Counter] | None = None):
        """Index raw documents, or pre-tokenised term-frequency Counters via `tf`."""
        self.k1, self.b = k1, b
        self.tf = tf if tf is not None else [Counter(tokenize(d)) for d in docs or []]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for c in self.tf for t in c)
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out

    def rank(self, query: str) -> list[int]:
        s = self.scores(query)
        return sorted(range(len(s)), key=lambda i: -s[i])
