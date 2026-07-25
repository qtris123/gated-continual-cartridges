#!/usr/bin/env bash
# Qasper QA -> MT benchmark on the open-ended parquet evals.
#
# Produces the requested table for Llama and Qwen:
#   icl_QA_raw
#   icl_MT_raw
#   icl_MT_plus_QA_cartridge
#   cartridge_p1
#   cartridge_p2
#
# Each condition is evaluated on both:
#   examples/qasper2/qasper_eval_QA.parquet
#   examples/qasper2/qasper_eval_MT.parquet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"

QA_EVAL="${QA_EVAL:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_QA.parquet}"
MT_EVAL="${MT_EVAL:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_MT.parquet}"
RUN_ROOT="${RUN_ROOT:-$CARTRIDGES_OUTPUT_DIR/qasper_loss_benchmark_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
DOWNLOAD_EVALS="${DOWNLOAD_EVALS:-1}"
MODELS="${MODELS:-llama qwen}"

ARGS=()
if [[ "$DOWNLOAD_EVALS" == "1" ]]; then
  ARGS+=(--download-evals)
fi
if [[ -n "${MAX_CONTEXT_TOKENS:-}" ]]; then
  ARGS+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
fi

echo "=========================================="
echo "Qasper Loss Benchmark — QA -> MT"
echo "=========================================="
echo "QA eval:   $QA_EVAL"
echo "MT eval:   $MT_EVAL"
echo "Output:    $RUN_ROOT"
echo "Models:    $MODELS"
echo "Device:    $DEVICE"
echo ""

source "$CARTRIDGES_DIR/.venv/bin/activate"

python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/qasper_loss_benchmark.py" \
  --qa-eval "$QA_EVAL" \
  --mt-eval "$MT_EVAL" \
  --output-dir "$RUN_ROOT" \
  --device "$DEVICE" \
  --models $MODELS \
  "${ARGS[@]}"

echo ""
echo "Done. Summary:"
echo "  $RUN_ROOT/qasper_loss_benchmark_summary.json"
