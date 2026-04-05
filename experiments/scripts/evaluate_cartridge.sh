#!/bin/bash
#SBATCH --job-name=eval-cartridge
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

AMD_P1="$EVAL_DATA/amd_eval_questions_phase1.parquet"
AMD_P2="$EVAL_DATA/amd_eval_questions_phase2.parquet"
#PEPSI_P1="$EVAL_DATA/amd_pepsi_eval_questions_phase1.parquet"
#PEPSI_P2="$EVAL_DATA/amd_pepsi_eval_questions_phase2.parquet"


MODEL="Qwen/Qwen3-4B-Instruct-2507"
INITIAL_CACHE="$OUTPUTS/2026-03-28-12-48-24-initial/7bd35e62-3a4a-47ed-a41c-2ef0db6b044f/cache_last.pt"
CONTINUAL_CACHE="$OUTPUTS/2026-03-28-19-07-49-continual/0828cdb8-8c62-4b81-8fae-e3d28d928d1e/cache_last.pt"
OUTPUT_DIR="$REPO_DIR/outputs/eval_amd_qwen3_4b_instruct/toks2048"

source "$REPO_DIR/.venv/bin/activate"
if [ -f "$REPO_DIR/.env" ]; then source "$REPO_DIR/.env"; fi

echo "=== Evaluating cartridge (task $SLURM_ARRAY_TASK_ID) ==="
echo "Model:           $MODEL"
echo "Initial cache:   $INITIAL_CACHE"
echo "Continual cache: $CONTINUAL_CACHE"
echo "Output dir:      $OUTPUT_DIR"
echo "============================"

ARGS=(
  --model "$MODEL"
  --initial-cache "$INITIAL_CACHE"
  --cache "$CONTINUAL_CACHE"
  --phase1-eval "$AMD_P1"
  --phase2-eval "$AMD_P2"
  --output-dir "$OUTPUT_DIR"
  --max-new-tokens 512
  --check-forgetting
)

python3 "$REPO_DIR/experiments/evaluate/cartridge.py" "${ARGS[@]}"
echo "=== Done ==="
