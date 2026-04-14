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

#PHASE1_EVAL="$EVAL_DATA/amd_eval_questions_phase1.parquet"
#PHASE1_EVAL="$EVAL_DATA/amd_eval_questions_phase2.parquet"
PHASE1_EVAL="$EVAL_DATA/amd_pepsi_eval_questions_phase1.parquet"
PHASE2_EVAL="$EVAL_DATA/amd_pepsi_eval_questions_phase2.parquet"


MODEL="meta-llama/Llama-3.2-3B-Instruct"
INITIAL_CACHE="$OUTPUTS/2026-03-24-02-12-18-train_initial/859804cb-7529-4500-9a4d-a0de831180d8/cache_last.pt"
CONTINUAL_CACHE="$OUTPUTS/2026-04-07-08-04-39-continual_frozen_layers/f620a3c7-754e-4560-a12a-41ce04746b3c/cache_last.pt"
OUTPUT_DIR="$REPO_DIR/outputs/eval_amd_llama_3.2_3b_instruct/toks512_frozen14-27_both"

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
  --phase1-eval "$PHASE1_EVAL"
  --phase2-eval "$PHASE2_EVAL"
  --output-dir "$OUTPUT_DIR"
  --max-new-tokens 512
  --check-forgetting
)

python3 "$REPO_DIR/experiments/evaluate/cartridge.py" "${ARGS[@]}"
echo "=== Done ==="
