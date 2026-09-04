#!/usr/bin/env bash
# Record free-form GENERATIONS (generations.jsonl) for one dataset+tag, fanning the
# 5 stages across GPUs. This is the autoregressive companion to the teacher-forced
# logppl eval: logppl stays the headline metric (matrix.json from the chain), while
# this pass captures the actual model text. Categories without a scorer
# (techqa_freeform / qasper_freeform) are recorded with score=null (record-only),
# so F1/anything can be computed later from generated_answer+reference_answer.
#
# Usage: record_generations.sh <dataset> <tag> <gpu_csv>
#   record_generations.sh techqa topt64_v1 0,1,2,3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$ROOT"
PY="${CARTRIDGES_PYTHON:-.venv/bin/python}"
export CARTRIDGES_DIR="$ROOT" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=1
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

DS="$1"; TAG="$2"; GPUS="${3:-0}"
PROTO=generations-freeform-v1
# continual_chain writes qasper state under qasper_asr_kg_state; the rest use *_5phase_state.
case "$DS" in
  qasper) STATE_DIR="outputs/qasper_asr_kg_state/${TAG}" ;;
  *)      STATE_DIR="outputs/${DS}_5phase_state/${TAG}" ;;
esac
OUT_DIR="outputs/evaluations/${DS}/${TAG}/${PROTO}"
PLAN="${OUT_DIR}/plan.json"

$PY examples/shared/evaluate/build_generation_matrix_plan.py \
  --dataset "$DS" --technique "$TAG" --model "$MODEL_NAME" \
  --state-dir "$STATE_DIR" --state-template 'p{phase:02d}.json' \
  --eval-template "data/${DS}/phases/phase{phase}_eval.parquet" \
  --output-dir "$OUT_DIR" --plan-out "$PLAN" \
  --phases 1 2 3 4 5 --batch-size 16 --max-new-tokens 256 --temperature 0.0

# Fan the five stages across the provided GPUs (avoids 5 model copies on one GPU).
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}
pids=(); i=0
for phase in 1 2 3 4 5; do
  sid=$(printf 'p%02d' "$phase")
  g=${GPU_ARR[$(( i % NG ))]}
  CUDA_VISIBLE_DEVICES="$g" $PY examples/shared/evaluate/generation_accuracy_matrix.py \
    --plan "$PLAN" --stages "$sid" \
    > "logs/gen_${DS}_${TAG}_${sid}.log" 2>&1 &
  pids+=($!); i=$(( i + 1 ))
done
for p in "${pids[@]}"; do wait "$p"; done

$PY examples/shared/evaluate/generation_accuracy_matrix.py --plan "$PLAN" --summarize-only >/dev/null
echo "=== generations recorded: ${OUT_DIR}/cells/*/generations.jsonl ==="
