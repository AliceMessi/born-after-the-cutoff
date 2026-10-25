"""Prepare and push one private kernel folder: copies kaggle/common.py next to its
run.py (kernels are single-file scripts, so common.py is inlined at the top) and writes
kernel-metadata.json with is_private=true.

  python -I kaggle/push.py b2_dense --gpu            (prints the command; add --push to run it)
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG_PREFIX = "agentloc-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--kernel-source", action="append", default=[], help="our own kernel whose output is mounted")
    ap.add_argument("--config", default="", help="JSON injected as CONFIG; its 'slug' names the kernel")
    args = ap.parse_args()
    src = HERE / args.kernel
    cfg = json.loads(args.config) if args.config else {}
    slug = SLUG_PREFIX + (cfg.get("slug") or args.kernel.replace("_", "-"))
    build = HERE.parent / "data" / "kaggle_kernels" / slug
    build.mkdir(parents=True, exist_ok=True)
    common = (HERE / "common.py").read_text(encoding="utf-8")
    run = (src / "run.py").read_text(encoding="utf-8")
    run = run.replace("from common import OUT, P, record_env, save_runs, setup, sh", "# (common.py inlined above)")
    header = f"CONFIG = {cfg!r}\n" if cfg else ""
    (build / "run.py").write_text(header + common + "\n\n# ---- kernel ----\n" + run, encoding="utf-8")
    meta = {
        "id": f"<kaggle-user>/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": args.gpu, "enable_internet": True,
        "dataset_sources": ["<kaggle-user>/agentloc-v01-private"], "competition_sources": [],
        "kernel_sources": [f"<kaggle-user>/{k}" for k in args.kernel_source], "model_sources": [],
    }
    if args.gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    assert meta["is_private"] is True and not meta["competition_sources"]
    cmd = ["kaggle", "kernels", "push", "-p", str(build)]
    print(" ".join(cmd))
    if args.push:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
