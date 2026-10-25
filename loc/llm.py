"""Minimal chat client for local model servers. stdlib only, so the same code
runs locally and in a Kaggle notebook.

Backends (AGENTLOC_LLM_BACKEND, or the `backend` argument):
  openai  OpenAI-compatible /chat/completions (LM Studio, llama.cpp, vLLM).
          `reasoning_effort="none"` turns Gemma 4 thinking off on LM Studio.
  ollama  Ollama's native /api/chat, which (unlike its OpenAI-compatible
          endpoint) accepts a per-request context size (`num_ctx`) and an explicit
          `think: false`. Ollama's default context would silently truncate our
          ~10k-token prompts, so `num_ctx` is always sent.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request


def _post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(messages: list[dict], model: str, base_url: str = "http://127.0.0.1:1234/v1",
         max_tokens: int = 512, temperature: float = 0.0, seed: int = 0, timeout: int = 900,
         reasoning_effort: str | None = "none", backend: str | None = None, num_ctx: int | None = None) -> dict:
    backend = backend or os.environ.get("AGENTLOC_LLM_BACKEND", "openai")
    t0 = time.time()
    if backend == "ollama":
        root = base_url.rstrip("/").removesuffix("/v1")
        out = _post(f"{root}/api/chat", {
            "model": model, "messages": messages, "stream": False,
            # "none" -> thinking off; None (the --thinking ablation) -> on, as LM Studio's default
            "think": reasoning_effort != "none",
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens,
                        "num_ctx": num_ctx or int(os.environ.get("AGENTLOC_NUM_CTX", "16384"))},
        }, timeout)
        msg = out.get("message") or {}
        return {
            "text": msg.get("content") or "",
            "prompt_tokens": out.get("prompt_eval_count"),
            "completion_tokens": out.get("eval_count"),
            "reasoning_tokens": None,
            "thinking_chars": len(msg.get("thinking") or ""),
            "truncated_prompt": None,
            "seconds": round(time.time() - t0, 2),
        }
    out = _post(f"{base_url}/chat/completions", {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "seed": seed, "stream": False,
        **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
    }, timeout)
    usage = out.get("usage") or {}
    return {
        "text": out["choices"][0]["message"].get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "seconds": round(time.time() - t0, 2),
    }
