#!/usr/bin/env bash
# Unattended, multi-GPU Qasper QA -> MT loss benchmark.
#
# Builds one job per (model_family, method) and runs them across all available
# GPUs (one job per GPU at a time, queuing the rest). Merges results into a
# single summary + table when done.
#
# Runs fully unattended. To launch in the background and detach:
#   nohup bash examples/qasper2/scripts/benchmarks/run_qasper_loss_benchmark_parallel.sh \
#     > outputs/qasper_loss_benchmark.out 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
QA_EVAL="${QA_EVAL:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_QA.parquet}"
MT_EVAL="${MT_EVAL:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_MT.parquet}"
RUN_ROOT="${RUN_ROOT:-$CARTRIDGES_OUTPUT_DIR/qasper_loss_benchmark_$(date +%Y%m%d_%H%M%S)}"
MODELS="${MODELS:-llama qwen}"

EXTRA_ARGS=()
if [[ -n "${GPUS:-}" ]]; then
  EXTRA_ARGS+=(--gpus $GPUS)
fi
if [[ -n "${MAX_CONTEXT_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
fi
if [[ -n "${PREFILL_CHUNK_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--prefill-chunk-size "$PREFILL_CHUNK_SIZE")
fi

echo "=========================================="
echo "Qasper Loss Benchmark (parallel dispatcher)"
echo "=========================================="
echo "Run root:  $RUN_ROOT"
echo "Models:    $MODELS"
echo "ICL ctx:   full topic panel (all papers in QA/MT stage)"
echo "QA eval:   $QA_EVAL"
echo "MT eval:   $MT_EVAL"
echo "Started:   $(date)"
echo ""

source "$CARTRIDGES_DIR/.venv/bin/activate"

python3 "$CARTRIDGES_DIR/examples/qasper2/experiments/dispatch_loss_benchmark.py" \
  --output-dir "$RUN_ROOT" \
  --qa-eval "$QA_EVAL" \
  --mt-eval "$MT_EVAL" \
  --models $MODELS \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Finished:  $(date)"
echo "Summary:   $RUN_ROOT/qasper_loss_benchmark_summary.json"
