#!/usr/bin/env bash
# Build + run the freeform-mc-options primeAnswer accuracy matrix for one lineage
# tag, fanning the 5 stages across GPUs. Matches the baseline protocol exactly
# (max_new_tokens=64, temp=0, answer_prime=" Answer:") so numbers are comparable
# across arms.
#
# Usage: run_accuracy.sh <dataset> <tag> <gpu_csv>
#   run_accuracy.sh quality soft_l1_v1 0,1,2,3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$ROOT"
if [[ -n "${CARTRIDGES_PYTHON:-}" ]]; then
  PY="$CARTRIDGES_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "$CONDA_PREFIX/bin/python" ]]; then
  PY="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/.conda/envs/rocky9/2024.09/cartridges/bin/python" ]]; then
  PY="$HOME/.conda/envs/rocky9/2024.09/cartridges/bin/python"
elif [[ -x "$HOME/.conda/envs/cartridges/bin/python" ]]; then
  PY="$HOME/.conda/envs/cartridges/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
export CARTRIDGES_DIR="$ROOT" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=1
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

DS="$1"; TAG="$2"; GPUS="${3:-0}"
PROTO=accuracy-freeform-mc-options-primeAnswer-v1
STATE_DIR="outputs/${DS}_5phase_state/${TAG}"
OUT_DIR="outputs/evaluations/${DS}/${TAG}/${PROTO}"
PLAN="${OUT_DIR}/plan.json"
mkdir -p logs "$OUT_DIR"

$PY examples/shared/evaluate/build_generation_matrix_plan.py \
  --dataset "$DS" --technique "$TAG" --model "$MODEL_NAME" \
  --state-dir "$STATE_DIR" --state-template 'p{phase:02d}.json' \
  --eval-template "data/${DS}/phases/phase{phase}_eval.parquet" \
  --output-dir "$OUT_DIR" --plan-out "$PLAN" \
  --phases 1 2 3 4 5 --batch-size 16 --max-new-tokens 64 --temperature 0.0 \
  --answer-prime ' Answer:'

# Fan the five stages across the provided GPUs.
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}
pids=()
i=0
for phase in 1 2 3 4 5; do
  sid=$(printf 'p%02d' "$phase")
  g=${GPU_ARR[$(( i % NG ))]}
  CUDA_VISIBLE_DEVICES="$g" $PY examples/shared/evaluate/generation_accuracy_matrix.py \
    --plan "$PLAN" --stages "$sid" \
    > "logs/acc_${DS}_${TAG}_${sid}.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
done
for p in "${pids[@]}"; do wait "$p"; done

# Summarize into matrix.json / matrix.csv.
$PY examples/shared/evaluate/generation_accuracy_matrix.py --plan "$PLAN" --summarize-only
echo "=== accuracy matrix done: ${OUT_DIR}/matrix.json ==="
