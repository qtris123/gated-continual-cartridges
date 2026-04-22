#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=30:00
#SBATCH --job-name=init_kvcache
#SBATCH --output=logs/init_kvcache_%j.out
#SBATCH --error=logs/init_kvcache_%j.err
set -euo pipefail


export CARTRIDGES_DIR=/home/vo43/cartridges

MODEL="qwen3-4b-instruct" # llama-3.2-3b-instruct
model_prefix = MODEL.split("-")[0]
TEXT="$CARTRIDGES_DIR/experiments_longhealth/longhealth_context.txt
OUTPUT_DIR="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt"

mkdir -p "$OUTPUT_DIR" logs

for TOKS in 512 1024 2048; do
  echo "=== Building init cache: model=${MODEL} toks=${TOKS} ==="
  python3 "$CARTRIDGES_DIR/experiments_longhealth/swapping/build_init_cache.py" \
    --model "$MODEL" \
    --text "$TEXT" \
    --max-tokens "$TOKS" \
    --output "$OUTPUT_DIR/${model_prefix}_${TOKS}_p1_cache_last.pt"
  echo "=== Done toks=${TOKS} ==="
done

echo "### All init caches built ###"