#!/bin/bash
#SBATCH --job-name=eval-amd2021-mcq
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
  REPO_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

OUTPUTS="$REPO_DIR/outputs"
EVAL_DATA="$REPO_DIR/data/financebench/eval"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
INITIAL_CACHE="$OUTPUTS/2026-03-24-09-31-09-train_initial/79909b2b-2a9c-407b-8b36-0db0f25f6f3f/cache_last.pt"
EVAL_PARQUET="$EVAL_DATA/amd_2021_120_batch-mcq.parquet"
OUTPUT_DIR="$REPO_DIR/outputs/eval_amd2021_llama_3.2_3b_instruct/toks512_mcq"

source "$REPO_DIR/.venv/bin/activate"
if [ -f "$REPO_DIR/.env" ]; then source "$REPO_DIR/.env"; fi

echo "=== Evaluating AMD 2021 MCQ (10 samples) ==="
echo "Model:      $MODEL"
echo "Cache:      $INITIAL_CACHE"
echo "Eval data:  $EVAL_PARQUET"
echo "Output dir: $OUTPUT_DIR"
echo "============================"

ARGS=(
  --model "$MODEL"
  --cache "$INITIAL_CACHE"
  --eval-data "$EVAL_PARQUET"
  --output-dir "$OUTPUT_DIR"
  --max-new-tokens 2048
  --eval-mode generate-score
  --max-samples 10
)

python3 "$REPO_DIR/experiments/evaluate/cartridge.py" "${ARGS[@]}"
echo "=== Done ==="
