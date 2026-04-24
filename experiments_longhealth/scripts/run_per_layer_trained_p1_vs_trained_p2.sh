#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --job-name=per_layer_trained_p1_vs_trained_p2
#SBATCH --output=per_layer_trained_p1_vs_trained_p2_%j.out
#SBATCH --error=per_layer_trained_p1_vs_trained_p2_%j.err
set -euo pipefail

export CARTRIDGES_DIR=/home/vo43/cartridges
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"

cd "$CARTRIDGES_DIR"

# per_layer_swap.py requires name=/path for each --eval (see parse_eval_args).
EVAL="longhealth=$CARTRIDGES_DIR/experiments_longhealth/longhealth_patient11_20_og.parquet"

declare -A TRAINED

for MODEL in "qwen3-4b-instruct" "llama-3.2-3b-instruct"; do
  model_prefix="${MODEL%%-*}"
  TRAINED=()
  TRAINED[512]="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_512_p2/cache_last.pt"
  TRAINED[1024]="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_1024_p2/cache_last.pt"
  TRAINED[2048]="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_2048_p2/cache_last.pt"
  for TOKS in 512 1024 2048; do
    P2="${TRAINED[$TOKS]}"
    P1="$CARTRIDGES_DIR/experiments_longhealth/cartridges_pt/${model_prefix}_${TOKS}_p1/cache_last.pt"

    echo "=== toks=${TOKS}  trained_p1_vs_trained_p2  per-layer sweep  model=${MODEL} ==="
    python3 "$CARTRIDGES_DIR/experiments_longhealth/swapping/per_layer_swap.py" \
      --model "$MODEL" \
      --mode openended \
      --cache-a "$P2" \
      --cache-b "$P1" \
      --eval "$EVAL" \
      --layers all \
      --directions A_into_B \
      --batch-size 32 \
      --output-dir "$CARTRIDGES_OUTPUT_DIR/per_layer_${model_prefix}_toks${TOKS}_trained_p1_vs_trained_p2_${OUT_TAG}"
  done
  echo "### DONE model: $MODEL"
done
