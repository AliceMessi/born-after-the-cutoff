"""Thin, cached wrapper around the GitHub CLI (`gh api`).

Uses the caller's existing `gh` login; never reads or prints tokens.
Every response is cached on disk (keyed by request) so a harvest can be
re-run offline and audited. Run scripts that import this with `python -I`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "gh"


def _key(*parts: str) -> Path:
    h = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]
    return CACHE / h[:2] / f"{h}.json"


_calls = {"graphql": 0, "core": 0}


def wait_budget(resource: str, min_remaining: int = 300) -> None:
    """Sleep until the rate-limit window resets if the budget is low.

    Checked every 25 live calls per resource (cache hits are free).
    """
    _calls[resource] += 1
    if _calls[resource] % 25:
        return
    p = subprocess.run(["gh", "api", "rate_limit"], capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        return
    res = json.loads(p.stdout)["resources"][resource]
    if res["remaining"] < min_remaining:
        pause = max(5, res["reset"] - int(time.time()) + 5)
        print(f"  [rate] {resource} remaining={res['remaining']}, sleeping {pause}s", flush=True)
        time.sleep(pause)


def _run(args: list[str], stdin: str | None = None, retries: int = 6) -> str:
    delay = 5.0
    for attempt in range(retries):
        p = subprocess.run(
            ["gh", "api", *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if p.returncode == 0:
            return p.stdout
        err = (p.stderr or "")[:400]
        # 404 / 422 are permanent; rate limits and 5xx are retried.
        if "HTTP 404" in err or "HTTP 422" in err or "Could not resolve to" in err:
            raise LookupError(err)
        if attempt == retries - 1:
            raise RuntimeError(f"gh api failed: {err}")
        if "rate limit" in err.lower():
            time.sleep(120)
            continue
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def graphql(query: str, variables: dict, use_cache: bool = True) -> dict:
    path = _key("graphql", query, json.dumps(variables, sort_keys=True))
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps({"query": query, "variables": variables})
    wait_budget("graphql")
    out = json.loads(_run(["graphql", "--input", "-"], stdin=payload))
    if out.get("errors"):
        raise RuntimeError(f"graphql errors: {json.dumps(out['errors'])[:400]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def rest(endpoint: str, raw: bool = False, use_cache: bool = True):
    """GET a REST endpoint. raw=True returns file bytes decoded as UTF-8 text."""
    path = _key("rest", endpoint, str(raw))
    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["text"] if raw else data
    wait_budget("core")
    args = [endpoint]
    if raw:
        args = ["-H", "Accept: application/vnd.github.raw", endpoint]
    text = _run(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw:
        path.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text
    data = json.loads(text)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data
