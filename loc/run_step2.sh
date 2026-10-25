#!/usr/bin/env bash
# Step 2 run chain (no Gemma LLM calls): one heavy stage at a time.
#   1. retrieval baselines R0/B1 on the evaluation ids
#   2. dense baseline B2 (EmbeddingGemma 2 via LM Studio), then unload
#   3. graph reachability diagnostics
#   4. evaluation tables
# Usage: bash loc/run_step2.sh   (from the project folder)
set -u
IDS=data/v0.1/eval_ids.txt
echo "[1/4] baselines $(date +%H:%M)"
python -I loc/run_baselines.py --ids-file "$IDS" 2>&1 | tail -2
echo "[2/4] dense B2 $(date +%H:%M)"
lms load text-embedding-embeddinggemma-2 --gpu max -y >/dev/null 2>&1
lms server start >/dev/null 2>&1
python -I loc/run_dense.py --model embeddinggemma-2 --served-id text-embedding-embeddinggemma-2 --ids-file "$IDS" 2>&1 | tail -2
lms unload --all >/dev/null 2>&1
lms server stop >/dev/null 2>&1
echo "[3/4] graph reachability $(date +%H:%M)"
python -I graph/reachability.py --ids-file "$IDS" 2>&1 | tail -16
echo "[4/4] evaluate $(date +%H:%M)"
python -I loc/evaluate.py 2>&1 | tail -3
echo "done $(date +%H:%M)"; du -sh data/repos data/cache
