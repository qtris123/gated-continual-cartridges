#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=30:00
#SBATCH --job-name=per_token_slot_init_vs_trained
#SBATCH --output=logs/per_token_slot_init_vs_trained_%j.out
#SBATCH --error=logs/per_token_slot_init_vs_trained_%j.err
set -euo pipefail

export CARTRIDGES_DIR=/home/vo43/cartridges
export CARTRIDGES_OUTPUT_DIR=/home/vo43/cartridges/outputs

cd "$CARTRIDGES_DIR"

EVAL="/home/vo43/cartridges/experiments_longhealth/longhealth_patient1_10_og.parquet"
MODEL="qwen3-4b-instruct" # llama-3.2-3b-instruct
model_prefix = $MODEL.split("-")[0]
declare -A TRAINED
TRAINED[512]="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_512_p1/cache_last.pt"
TRAINED[1024]="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_1024_p1/cache_last.pt"
TRAINED[2048]="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_2048_p1/cache_last.pt"

for TOKS in "$@"; do
  TR="${TRAINED[$TOKS]}"
  UNTR="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_${TOKS}_p1_cache_last.pt"

  echo "=== toks=${TOKS}  A_into_B (sufficiency: trained V into untrained base) ==="
  python3 $CARTRIDGES_DIR/experiments_longhealth/swapping/per_slot_swap.py \
    --model $MODEL \
    --cache-a "$TR" \
    --cache-b "$UNTR" \
    --eval "$EVAL" \
    --batch-size 32 \
    --output-dir "$CARTRIDGES_OUTPUT_DIR/per_token_slot_${model_prefix}_toks${TOKS}_init_vs_trained_A_into_B"

done
echo "### DONE sizes: $@"
