"""Localization metrics. Two Acc@k conventions are reported because the
literature uses both: LocAgent/SweRank 'all gold in top-k' and
IssueLoc/SemNav-style 'any gold in top-k' (Hit@k)."""
from __future__ import annotations

import math


def acc_all(pred: list[str], gold: set[str], k: int) -> float:
    return float(bool(gold) and gold <= set(pred[:k]))


def hit_any(pred: list[str], gold: set[str], k: int) -> float:
    return float(bool(gold & set(pred[:k])))


def mrr(pred: list[str], gold: set[str]) -> float:
    for i, p in enumerate(pred, 1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg(pred: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(pred[:k], 1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def report(pred: list[str], gold: set[str], ks=(1, 3, 5, 10)) -> dict:
    out = {"mrr": round(mrr(pred, gold), 4)}
    for k in ks:
        out[f"acc_all@{k}"] = acc_all(pred, gold, k)
        out[f"hit@{k}"] = hit_any(pred, gold, k)
        out[f"ndcg@{k}"] = round(ndcg(pred, gold, k), 4)
    return out
