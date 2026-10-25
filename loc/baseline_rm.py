"""Baseline RM: a repository map ranked by personalised PageRank (no LLM).

Inspired by aider's repo-map (https://aider.chat/docs/repomap.html), which ranks
definitions with PageRank over a reference graph. Here the graph is our own code
graph (graph/build_graph.py) and the personalisation vector is seeded by code-like
identifiers in the issue text. It measures what the graph alone contributes,
without any model.

Seeding: tokens that look like code (snake_case, CamelCase, dotted names,
backticked spans, *.py paths) are matched against node names (last dotted
component, dotted suffixes, file paths). A token matching m nodes gives each of
them weight 1/m, so ambiguous names count less. With no seed the ranking falls
back to plain PageRank (a structural prior).
"""
from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx

EDGE_WEIGHT = {"calls": 1.0, "inherits": 1.0, "contains": 0.5, "imports": 0.3}
_CODE_TOKEN = re.compile(r"`([^`\n]{2,80})`|([A-Za-z_][\w./]*\.py)|([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)|([A-Za-z_]\w{2,})")


def _looks_like_code(tok: str) -> bool:
    return "_" in tok or "." in tok or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok))


def issue_identifiers(text: str) -> list[str]:
    out = []
    for m in _CODE_TOKEN.finditer(text or ""):
        tick, path, dotted, word = m.groups()
        if tick:
            out += re.findall(r"[A-Za-z_][\w.]*", tick)
        elif path or dotted:
            out.append(path or dotted)
        elif word and _looks_like_code(word):
            out.append(word)
    return [t.strip(".") for t in out if len(t.strip(".")) > 2]


def rank_repo_map(graph: dict, issue: str, alpha: float = 0.85) -> tuple[list[str], list[str], int]:
    """Return (ranked files, ranked node ids, number of seeded nodes)."""
    g = nx.Graph()
    for n in graph["nodes"]:
        g.add_node(n["id"])
    for e in graph["edges"]:
        w = EDGE_WEIGHT.get(e["type"], 0.3)
        if g.has_edge(e["source"], e["target"]):
            g[e["source"]][e["target"]]["weight"] += w
        else:
            g.add_edge(e["source"], e["target"], weight=w)
    by_last = defaultdict(list)
    by_file = defaultdict(list)
    for n in graph["nodes"]:
        by_last[n["id"].rsplit(".", 1)[-1]].append(n["id"])
        by_file[n["file"]].append(n["id"])
    seeds: dict[str, float] = defaultdict(float)
    for tok in issue_identifiers(issue):
        if tok.endswith(".py"):
            hits = [i for f, ids in by_file.items() if f == tok or f.endswith("/" + tok) for i in ids
                    if graph_node_type(graph, i) == "module"]
        elif "." in tok:
            hits = [i for i in g.nodes if i == tok or i.endswith("." + tok)]
        else:
            hits = by_last.get(tok, [])
        for h in hits:
            seeds[h] += 1.0 / len(hits)
    if g.number_of_nodes() == 0:
        return [], [], 0
    pers = dict(seeds) if seeds else None
    scores = nx.pagerank(g, alpha=alpha, personalization=pers, weight="weight", max_iter=200, tol=1e-6)
    order = sorted(scores, key=lambda i: -scores[i])
    file_of = {n["id"]: n["file"] for n in graph["nodes"]}
    files = list(dict.fromkeys(file_of[i] for i in order))
    return files, order, len(seeds)


_TYPES: dict[int, dict[str, str]] = {}


def graph_node_type(graph: dict, node_id: str) -> str:
    key = id(graph)
    if key not in _TYPES:
        _TYPES.clear()
        _TYPES[key] = {n["id"]: n["type"] for n in graph["nodes"]}
    return _TYPES[key].get(node_id, "")
