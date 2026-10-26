"""Build the PRIVATE Kaggle working dataset `<owner>/agentloc-v01-private`.

Contents (our own harvest only; no Competition Data):
  project.zip  harvest/, loc/, graph/ code, plus data/v0.1/{gold.jsonl, eval_ids.txt,
               pilot_ids.txt, audit_exclusions.txt, runs/B1-bm25.jsonl} and
               data/v0.1/raw/texts.candidates.jsonl (issue title/body per instance)

  python -I kaggle/build_dataset.py           -> data/kaggle_ds/agentloc-v01-private/
  kaggle datasets create  -p data/kaggle_ds/agentloc-v01-private        (first time; private by default)
  kaggle datasets version -p data/kaggle_ds/agentloc-v01-private -m msg (updates)
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from owner import owner  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loc"))
from data import V01, read_jsonl  # noqa: E402

SLUG = "agentloc-v01-private"
OUT = ROOT / "data" / "kaggle_ds" / SLUG


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    texts = {}
    for f in (V01 / "raw").glob("*.candidates.jsonl"):
        for r in read_jsonl(f):
            texts[r["instance_id"]] = {"instance_id": r["instance_id"], "issue_title": r["issue_title"],
                                       "issue_body": r["issue_body"]}
    gold_ids = {r["instance_id"] for r in read_jsonl(V01 / "gold.jsonl")}
    text_lines = [json.dumps(texts[i], ensure_ascii=False) for i in sorted(gold_ids) if i in texts]
    with zipfile.ZipFile(OUT / "project.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("harvest", "loc", "graph"):
            for p in sorted((ROOT / sub).glob("*.py")) + sorted((ROOT / sub).glob("*.json")):
                z.write(p, f"project/{sub}/{p.name}")
        for name in ("gold.jsonl", "eval_ids.txt", "pilot_ids.txt", "pilot_g2_ids.txt", "g2_ids.txt", "main_ids.txt", "audit_exclusions.txt"):
            z.write(V01 / name, f"project/data/v0.1/{name}")
        b1 = V01 / "runs" / "B1-bm25.jsonl"
        if b1.exists():
            z.write(b1, "project/data/v0.1/runs/B1-bm25.jsonl")
        z.writestr("project/data/v0.1/raw/texts.candidates.jsonl", chr(10).join(text_lines) + chr(10))
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": SLUG,
        "id": f"{owner()}/{SLUG}",
        "licenses": [{"name": "other"}],
        "subtitle": "Private working set for a localization benchmark (own GitHub harvest)",
    }, indent=1), encoding="utf-8")
    size = (OUT / "project.zip").stat().st_size / 1e6
    print(f"{OUT / 'project.zip'}: {size:.1f} MB, {len(text_lines)} issue texts, {len(gold_ids)} gold records")


if __name__ == "__main__":
    main()
