"""Minimal OpenAI-compatible chat client (LM Studio / llama.cpp / vLLM).

stdlib only, so the same code runs locally and in a Kaggle notebook.
"""
from __future__ import annotations

import json
import time
import urllib.request


def chat(messages: list[dict], model: str, base_url: str = "http://127.0.0.1:1234/v1",
         max_tokens: int = 512, temperature: float = 0.0, seed: int = 0, timeout: int = 600,
         reasoning_effort: str | None = "none") -> dict:
    """reasoning_effort="none" disables Gemma 4 thinking on LM Studio; None omits the field."""
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "seed": seed, "stream": False,
        **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    usage = out.get("usage") or {}
    return {
        "text": out["choices"][0]["message"].get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "seconds": round(time.time() - t0, 2),
    }
