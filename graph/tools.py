"""Graph tools with the same signatures as the competition harness:

  get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str
  search_similar_code(query: str, k: int = 10) -> str
  get_code_subgraph(nodes: list[str]) -> str

Signatures follow the competition's public "Model Selection, Budget, and
Harness Rules" text. The *return-string format* of the official tools is not
public without joining the main track, so ours return compact JSON.

search_similar_code uses precomputed node embeddings when given (cosine,
like the competition's 256-d embeddings); otherwise it falls back to BM25
over node ids and text, and says so in the result.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "loc"))
from bm25 import BM25  # noqa: E402


class CodeGraph:
    def __init__(self, graph: dict, embeddings: dict | None = None, query_encoder=None):
        """graph: node-link dict; embeddings: {node_id: list[float]} (unit-normalised);
        query_encoder: callable(str) -> list[float] for search_similar_code."""
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        for e in graph["edges"]:
            self.out[e["source"]].append((e["type"], e["target"]))
            self.inc[e["target"]].append((e["type"], e["source"]))
        self.emb = embeddings
        self.encode = query_encoder
        self._bm25 = None
        self._ids = list(self.nodes)

    # -- helpers
    def _brief(self, nid: str) -> dict:
        n = self.nodes[nid]
        return {"id": nid, "kind": n.get("type"), "file": n.get("file"),
                "lines": [n.get("start_line"), n.get("end_line")]}

    def resolve(self, node: str) -> str | None:
        """Exact id, else a unique suffix match (models often drop the package prefix)."""
        if node in self.nodes:
            return node
        hits = [i for i in self._ids if i.endswith("." + node)]
        return hits[0] if len(hits) == 1 else None

    # -- competition tools
    def get_code_neighbors(self, node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str:
        nid = self.resolve(node)
        if nid is None:
            return json.dumps({"error": f"unknown node: {node}"})
        outs = [(t, x) for t, x in self.out[nid] if edge_type in (None, t)]
        ins = [(t, x) for t, x in self.inc[nid] if edge_type in (None, t)]
        half = max(1, max_neighbors // 2)
        return json.dumps({
            "node": self._brief(nid),
            "outgoing": [{"edge": t, **self._brief(x)} for t, x in outs[:half]],
            "incoming": [{"edge": t, **self._brief(x)} for t, x in ins[: max_neighbors - min(len(outs), half)]],
            "truncated": len(outs) + len(ins) > max_neighbors,
        })

    def search_similar_code(self, query: str, k: int = 10) -> str:
        if self.emb is not None and self.encode is not None:
            q = self.encode(query)
            scored = sorted(((sum(a * b for a, b in zip(q, v)), i) for i, v in self.emb.items()), reverse=True)
            mode = "embedding"
        else:
            if self._bm25 is None:
                self._bm25 = BM25([i + "\n" + self.nodes[i].get("text", "") for i in self._ids])
            s = self._bm25.scores(query)
            scored = sorted(((s[j], self._ids[j]) for j in range(len(s))), reverse=True)
            mode = "bm25-fallback"
        return json.dumps({"mode": mode, "results": [{"score": round(sc, 4), **self._brief(i)}
                                                     for sc, i in scored[:k]]})

    def get_code_subgraph(self, nodes: list[str]) -> str:
        ids = [r for r in (self.resolve(n) for n in nodes) if r]
        keep = set(ids)
        edges = [{"source": s, "target": t, "edge": ty}
                 for s in ids for ty, t in self.out[s] if t in keep]
        return json.dumps({"nodes": [self._brief(i) for i in ids], "edges": edges,
                           "unknown": [n for n in nodes if self.resolve(n) is None]})
