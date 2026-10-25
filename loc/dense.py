"""Dense retrieval (B2) through an OpenAI-compatible /v1/embeddings endpoint
(LM Studio or llama.cpp server locally, Ollama on Kaggle). stdlib only.

Models (GGUF, official ggml-org conversions, Apache-2.0):
  embeddinggemma-2   EmbeddingGemma 2 (270M text backbone), Matryoshka -> 256-d,
                     matching the competition's 256-d node embeddings
  jina-code          jina-embeddings-v2-base-code (161M), native 768-d (no MRL truncation)

Vectors are cached in data/cache/emb.sqlite keyed by (model, sha256(input text)),
so unchanged functions are embedded once across all snapshots of a repository.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import os
import sqlite3
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("AGENTLOC_EMB_CACHE") or Path(__file__).resolve().parent.parent / "data" / "cache" / "emb.sqlite")
UNIT_CHARS = 1500  # ~400 tokens per unit keeps the GPU cost bounded

MODELS = {
    "embeddinggemma-2": {
        "dim": 256,
        "query": "task: code retrieval | query: {q}",
        "doc": "title: {title} | text: {text}",
    },
    "jina-code": {"dim": None, "query": "{q}", "doc": "{title}\n{text}"},
}


class Encoder:
    def __init__(self, name: str, served_id: str, base_url: str = "http://127.0.0.1:1234/v1", batch: int = 64):
        self.name, self.served_id, self.base_url, self.batch = name, served_id, base_url, batch
        self.cfg = MODELS[name]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, key TEXT, vec BLOB, PRIMARY KEY (model, key))")
        self.calls = 0

    def _post(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.served_id, "input": texts}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))["data"]
        self.calls += 1
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    def _finish(self, v: list[float]) -> list[float]:
        dim = self.cfg["dim"]
        if dim:
            v = v[:dim]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        out: dict[str, list[float]] = {}
        for i in range(0, len(keys), 900):  # sqlite parameter limit
            chunk = keys[i : i + 900]
            q = f"SELECT key, vec FROM emb WHERE model=? AND key IN ({','.join('?' * len(chunk))})"
            for k, blob in self.db.execute(q, [self.name, *chunk]):
                out[k] = array.array("f", blob).tolist()
        todo = [(k, t) for k, t in zip(keys, texts) if k not in out]
        seen = set()
        todo = [(k, t) for k, t in todo if not (k in seen or seen.add(k))]
        for i in range(0, len(todo), self.batch):
            part = todo[i : i + self.batch]
            vecs = [self._finish(v) for v in self._post([t for _, t in part])]
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?)",
                                [(self.name, k, array.array("f", v).tobytes()) for (k, _), v in zip(part, vecs)])
            out.update({k: v for (k, _), v in zip(part, vecs)})
        self.db.commit()
        return [out[k] for k in keys]

    def query(self, q: str) -> list[float]:
        return self.embed([self.cfg["query"].format(q=q[:6000])])[0]

    def doc(self, title: str, text: str) -> str:
        return self.cfg["doc"].format(title=title, text=text[:UNIT_CHARS])


class STEncoder(Encoder):
    """Same prompts, truncation and cache, but a local sentence-transformers model
    (used on Kaggle GPUs; e.g. `google/embeddinggemma-2`). Inputs are already
    formatted with the model's task prompts, so `prompt=""` disables any default."""

    def __init__(self, name: str, hf_id: str, device: str = "cuda", batch: int = 16, dtype: str = "float32",
                 max_seq_length: int = 512):
        super().__init__(name, served_id=hf_id, batch=batch)
        import torch  # noqa: F401  (only needed on the GPU host)
        from sentence_transformers import SentenceTransformer

        kwargs = {"torch_dtype": getattr(torch, dtype)}
        self.model = SentenceTransformer(hf_id, device=device, model_kwargs=kwargs)
        # units are capped at UNIT_CHARS (~400 tokens); the default 8k window OOMs a T4 at batch 64
        self.model.max_seq_length = max_seq_length

    def _post(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vecs = self.model.encode(texts, prompt="", batch_size=self.batch, convert_to_numpy=True,
                                 normalize_embeddings=False, show_progress_bar=False)
        return [v.tolist() for v in vecs]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
