"""Kaggle account that owns the kernels and datasets.

Read from the KAGGLE_OWNER environment variable, else from kaggle/owner.local (one line,
git-ignored), so no account name is hard-coded in the published code.
"""
from __future__ import annotations

import os
from pathlib import Path


def owner() -> str:
    name = os.environ.get("KAGGLE_OWNER", "").strip()
    local = Path(__file__).resolve().parent / "owner.local"
    if not name and local.exists():
        name = local.read_text(encoding="utf-8").strip()
    if not name:
        raise SystemExit("Set KAGGLE_OWNER or write your Kaggle username to kaggle/owner.local")
    return name
