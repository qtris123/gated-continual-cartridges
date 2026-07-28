#!/usr/bin/env bash
# Full benchmark eval: ICL baselines + cartridge cross-eval for comparison.
#
# Produces a unified comparison table showing:
#   - ICL stage-1 (QA context, raw model)
#   - ICL stage-2 (MT context + P1 cartridge)
#   - Cartridge-only (P1 and P2 cartridges, compressed context)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
DATA_DIR="${DATA_DIR:-$CARTRIDGES_DIR/examples/qasper2}"
HF_NS="${HF_NS:-qtris123}"
RUN_ROOT="${RUN_ROOT:-${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}/benchmark_eval_$(date +%Y%m%d_%H%M%S)}"
BACKEND="${BACKEND:-local}"
DEVICE="${DEVICE:-cuda}"
COT="${COT:-0}"
SIZE="${SIZE:-512}"

mkdir -p "$RUN_ROOT"

COMMON_ARGS=()
[[ "$COT" == "1" ]] && COMMON_ARGS+=(--cot)

DATASETS=(
  "$DATA_DIR/qasper_QA-mcq.csv:mcq"
  "$DATA_DIR/qasper_MT-mcq.csv:mcq"
  "$DATA_DIR/qasper_QA-yes-no.csv:yes_no"
  "$DATA_DIR/qasper_MT-yes-no.csv:yes_no"
)

source "$CARTRIDGES_DIR/.venv/bin/activate"

echo "=========================================="
echo "Full Benchmark Eval — Qasper QA -> MT"
echo "=========================================="
echo "RUN_ROOT=$RUN_ROOT"
echo "SIZE=$SIZE"
echo ""

# --- Models and cartridges ---
declare -A MODELS=(
  [llama]="meta-llama/Llama-3.2-3B-Instruct"
  [qwen]="Qwen/Qwen3-4B-Instruct-2507"
)
declare -A P1_CARTRIDGES=(
  [llama]="${HF_NS}/llama_qasper-QA-task_8192_${SIZE}_no-cartridge_10-epochs"
  [qwen]="${HF_NS}/qwen_qasper-QA-task_8192_${SIZE}_no-cartridge_10-epochs"
)
declare -A P2_CARTRIDGES=(
  [llama]="${HF_NS}/llama_qasper-MT-task_8192_10-epochs_with-cartridge_qasper-QA-task_8192_${SIZE}"
  [qwen]="${HF_NS}/qwen_qasper-MT-task_8192_10-epochs_with-cartridge_qasper-QA-task_8192_${SIZE}"
)

ICL_SUMMARIES=()
CART_SUMMARIES=()

for family in llama qwen; do
  model="${MODELS[$family]}"
  p1="${P1_CARTRIDGES[$family]}"
  p2="${P2_CARTRIDGES[$family]}"

  # --- ICL: stage-1 (QA context, raw model) ---
  icl1_dir="$RUN_ROOT/${family}${SIZE}_stage1_icl"
  mkdir -p "$icl1_dir"
  echo "--- ICL stage-1: $family ---"
  python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/icl_eval.py" \
    --context-topic QA \
    --model "$model" \
    --output-dir "$icl1_dir" \
    --device "$DEVICE" \
    "${COMMON_ARGS[@]}" \
    --datasets "${DATASETS[@]}"
  ICL_SUMMARIES+=("$icl1_dir/icl_eval_summary.json")

  # --- ICL: stage-2 (MT context + P1 cartridge) ---
  icl2_dir="$RUN_ROOT/${family}${SIZE}_stage2_icl_plus_p1"
  mkdir -p "$icl2_dir"
  echo "--- ICL stage-2 + P1 cartridge: $family ---"
  python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/icl_eval.py" \
    --context-topic MT \
    --cartridges "$p1" \
    --model "$model" \
    --output-dir "$icl2_dir" \
    --device "$DEVICE" \
    "${COMMON_ARGS[@]}" \
    --datasets "${DATASETS[@]}"
  ICL_SUMMARIES+=("$icl2_dir/icl_eval_summary.json")

  # --- Cartridge cross-eval (P1 and P2) ---
  cart_dir="$RUN_ROOT/${family}${SIZE}_cartridge_cross"
  mkdir -p "$cart_dir"
  echo "--- Cartridge cross-eval: $family ---"
  python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/cartridge_eval.py" \
    --backend "$BACKEND" \
    --cartridges "$p1" "$p2" \
    --model "$model" \
    --output-dir "$cart_dir" \
    --device "$DEVICE" \
    "${COMMON_ARGS[@]}" \
    --datasets "${DATASETS[@]}"
  CART_SUMMARIES+=("$cart_dir/cartridge_eval_summary.json")
done

# --- Merge all results ---
python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/compare_benchmark_results.py" \
  --icl-summaries "${ICL_SUMMARIES[@]}" \
  --cartridge-summaries "${CART_SUMMARIES[@]}" \
  --output "$RUN_ROOT/benchmark_comparison.json"

echo ""
echo "=========================================="
echo "Benchmark eval complete!"
echo "Results: $RUN_ROOT"
echo "Comparison: $RUN_ROOT/benchmark_comparison.json"
echo "=========================================="
