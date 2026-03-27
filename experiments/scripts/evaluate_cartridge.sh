#!/bin/bash
#SBATCH --job-name=eval-cartridge
#SBATCH --gres=gpu:1
#SBATCH --array=1-6
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

PEPSI_P1="$EVAL_DATA/amd_pepsi_eval_questions_phase1.parquet"
PEPSI_P2="$EVAL_DATA/amd_pepsi_eval_questions_phase2.parquet"

case "$SLURM_ARRAY_TASK_ID" in
  # ── Llama | AMD2021 → Pepsi2021 ───────────────────────────────────────────
  1)
    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    INITIAL_CACHE="$OUTPUTS/2026-03-24-02-12-18-train_initial/859804cb-7529-4500-9a4d-a0de831180d8/cache_last.pt"
    CONTINUAL_CACHE="$OUTPUTS/2026-03-24-03-31-33-train_continual/c4064a62-4130-4c05-a249-08160ef24e00/cache_last.pt"
    OUTPUT_DIR="$REPO_DIR/outputs/eval_pepsi_llama/toks512"
    ;;
  2)
    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    INITIAL_CACHE="$OUTPUTS/2026-03-24-00-51-52-train_initial/bf6965fb-7360-49e0-b63d-3a144927500b/cache_last.pt"
    CONTINUAL_CACHE="$OUTPUTS/2026-03-24-01-10-06-train_continual/1586e9f3-3b55-438b-9682-c16a9d9c158e/cache_last.pt"
    OUTPUT_DIR="$REPO_DIR/outputs/eval_pepsi_llama/toks1024"
    ;;
  3)
    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    INITIAL_CACHE="$OUTPUTS/2026-03-24-02-26-39-train_initial/89363b00-cdf7-4ba9-9e02-336358b18f99/cache_last.pt"
    CONTINUAL_CACHE="$OUTPUTS/2026-03-24-03-44-52-train_continual/3f9bcfa8-df6c-490f-91a9-fc7815eb40d8/cache_last.pt"
    OUTPUT_DIR="$REPO_DIR/outputs/eval_pepsi_llama/toks2048"
    ;;
  # ── Qwen | AMD2021 → Pepsi2021 ────────────────────────────────────────────
  4)
    MODEL="Qwen/Qwen3-4B"
    INITIAL_CACHE="$OUTPUTS/2026-03-24-08-05-36-train_initial/925fa9f2-629c-437d-a2e7-37c8f51febe2/cache_last.pt"
    CONTINUAL_CACHE="$OUTPUTS/2026-03-24-10-11-43-train_continual/c007c5c7-eacc-4b7c-b8e8-8c708919098a/cache_last.pt"
    OUTPUT_DIR="$REPO_DIR/outputs/eval_pepsi_qwen/toks512"
    ;;
  5)
    MODEL="Qwen/Qwen3-4B"
    INITIAL_CACHE="$OUTPUTS/2026-03-24-08-32-47-train_initial/97d3c0d0-c33b-4754-9f56-1fc3cd22d04a/cache_last.pt"
    CONTINUAL_CACHE="$OUTPUTS/2026-03-24-10-21-42-train_continual/0a179602-a4cf-41d5-9163-48376c8741d3/cache_last.pt"
    OUTPUT_DIR="$REPO_DIR/outputs/eval_pepsi_qwen/toks1024"
    ;;
  6)
    MODEL="Qwen/Qwen3-4B"
    INITIAL_CACHE="$OUTPUTS/2026-03-24-09-01-04-train_initial/be39b7e3-1e7e-477b-b4c8-46cf60a010a3/cache_last.pt"
    CONTINUAL_CACHE="$OUTPUTS/2026-03-24-10-29-13-train_continual/22ee641a-358e-445e-a8e0-718d53adb445/cache_last.pt"
    OUTPUT_DIR="$REPO_DIR/outputs/eval_pepsi_qwen/toks2048"
    ;;
  *)
    echo "Error: unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

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
  --phase1-eval "$PEPSI_P1"
  --phase2-eval "$PEPSI_P2"
  --output-dir "$OUTPUT_DIR"
  --max-new-tokens 512
  --check-forgetting
)

python3 "$REPO_DIR/experiments/evaluate/cartridge.py" "${ARGS[@]}"
echo "=== Done ==="
