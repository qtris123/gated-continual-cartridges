#!/usr/bin/env bash
# Run ICL baseline evaluations for Qasper QA -> MT benchmark.
#
# Conditions:
#   1. stage1_icl          — QA full context + raw model
#   2. stage2_icl_plus_p1  — MT full context + Phase-1 cartridge
#
# Evaluates both QA and MT datasets (MCQ and yes/no) for Llama and Qwen 512.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
DATA_DIR="${DATA_DIR:-$CARTRIDGES_DIR/examples/qasper2}"
HF_NS="${HF_NS:-qtris123}"
RUN_ROOT="${RUN_ROOT:-${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}/icl_baseline_$(date +%Y%m%d_%H%M%S)}"
BACKEND="${BACKEND:-local}"
DEVICE="${DEVICE:-cuda}"
COT="${COT:-0}"
TOKA_PORT="${TOKA_PORT:-10210}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "$RUN_ROOT"

COMMON_ARGS=()
[[ "$COT" == "1" ]] && COMMON_ARGS+=(--cot)
[[ "$BACKEND" == "tokasaurus" ]] && COMMON_ARGS+=(--backend tokasaurus --url "http://$(hostname):${TOKA_PORT}")

DATASETS=(
  "$DATA_DIR/qasper_QA-mcq.csv:mcq"
  "$DATA_DIR/qasper_MT-mcq.csv:mcq"
  "$DATA_DIR/qasper_QA-yes-no.csv:yes_no"
  "$DATA_DIR/qasper_MT-yes-no.csv:yes_no"
)

run_icl() {
  local label="$1"
  local model="$2"
  local context_topic="$3"
  local cartridge="${4:-}"

  local out_dir="$RUN_ROOT/${label}"
  mkdir -p "$out_dir"

  echo ""
  echo "══════════════════════════════════════════"
  echo "  $label"
  echo "  Model:         $model"
  echo "  Context topic: $context_topic"
  echo "  Cartridge:     ${cartridge:-none}"
  echo "  Output:        $out_dir"
  echo "══════════════════════════════════════════"

  local args=(
    python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/icl_eval.py"
    --context-topic "$context_topic"
    --model "$model"
    --output-dir "$out_dir"
    --device "$DEVICE"
    "${COMMON_ARGS[@]}"
  )

  for ds in "${DATASETS[@]}"; do
    args+=(--datasets "$ds")
  done

  if [[ -n "$cartridge" ]]; then
    args+=(--cartridges "$cartridge")
  fi

  "${args[@]}"
}

echo "=========================================="
echo "ICL Baseline Benchmark — Qasper QA -> MT"
echo "=========================================="
echo "RUN_ROOT=$RUN_ROOT"
echo "BACKEND=$BACKEND"
echo ""

source "$CARTRIDGES_DIR/.venv/bin/activate"

# --- Llama 512 ---
LLAMA_MODEL="meta-llama/Llama-3.2-3B-Instruct"
LLAMA_P1="${HF_NS}/llama_qasper-QA-task_8192_512_no-cartridge_10-epochs"

run_icl "llama512_stage1_icl" "$LLAMA_MODEL" "QA"
run_icl "llama512_stage2_icl_plus_p1" "$LLAMA_MODEL" "MT" "$LLAMA_P1"

# --- Qwen 512 ---
QWEN_MODEL="Qwen/Qwen3-4B-Instruct-2507"
QWEN_P1="${HF_NS}/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs"

run_icl "qwen512_stage1_icl" "$QWEN_MODEL" "QA"
run_icl "qwen512_stage2_icl_plus_p1" "$QWEN_MODEL" "MT" "$QWEN_P1"

echo ""
echo "=========================================="
echo "ICL baseline eval complete!"
echo "Results: $RUN_ROOT"
echo "=========================================="

# Merge summaries
python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/compare_benchmark_results.py" \
  --icl-summaries \
    "$RUN_ROOT/llama512_stage1_icl/icl_eval_summary.json" \
    "$RUN_ROOT/llama512_stage2_icl_plus_p1/icl_eval_summary.json" \
    "$RUN_ROOT/qwen512_stage1_icl/icl_eval_summary.json" \
    "$RUN_ROOT/qwen512_stage2_icl_plus_p1/icl_eval_summary.json" \
  --output "$RUN_ROOT/icl_baseline_comparison.json"

echo "Comparison table: $RUN_ROOT/icl_baseline_comparison.json"
