#!/usr/bin/env bash
# QuALITY replication of the QASPER 5x5 continual-AM grid.
#
# Stage 1 is a fresh RoPE-corrected AM compaction of QuALITY phase 1 (the arm-D
# settings: rebake the selected keys at the model's own rotary base with global
# teacher positions, beta off). Stages 2-5 then run the same continual write
# recipe as the QASPER lineage, through the same driver, so the only thing that
# differs between the two grids is the data.
#
# Both steps are resumable: the compaction is skipped if its cache exists, and
# each continual stage is guarded by a marker holding all five metrics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_DIR="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

TAG="${TAG:-delta_ha_b0_idf0_ropefix_p01D}"
GPU="${GPU:-0}"
P1_ROOT="${P1_ROOT:-$ROOT/outputs/quality_p1_ropefix}"
PY_BIN="${CARTRIDGES_PYTHON:-${PY:-python}}"
export CARTRIDGES_PYTHON="$PY_BIN"

# --- stage 1: RoPE-corrected AM compaction of QuALITY phase 1 -----------------
P01_CACHE="${P01_CACHE:-}"
if [ -z "$P01_CACHE" ]; then
  P01_CACHE="$(ls -1 "$P1_ROOT"/*/*/cache_last.pt 2>/dev/null | head -1 || true)"
fi

if [ -n "$P01_CACHE" ] && [ -f "$P01_CACHE" ]; then
  echo "=== p01 compaction already present: $P01_CACHE"
else
  echo "=== p01: AM compaction of QuALITY phase 1 (arm-D settings) ==="
  mkdir -p "$P1_ROOT"
  for phase in 1 2 3 4 5; do
    export "EVAL_P${phase}_PATH=$ROOT/data/quality/phases/phase${phase}_eval.parquet"
    export "EVAL_P${phase}_NAME=p${phase}"
  done
  CUDA_VISIBLE_DEVICES="$GPU" \
  CARTRIDGES_OUTPUT_DIR="$P1_ROOT" \
  AM_DATASET=quality \
  AM_QUALITY_PHASE=1 \
  QA_DATA_PATH="$ROOT/data/quality/train/qwen_quality_p1_task_8192.parquet" \
  NUM_TOKENS="${NUM_TOKENS:-512}" \
  KEY_SELECT=highest_attention \
  GRANULARITY="${P1_GRANULARITY:-per_head}" \
  ENABLE_BETA=0 \
  REBAKE_KEY_POSITIONS=1 \
  AM_ROPE_THETA=model \
  GLOBAL_TEACHER_POSITIONS=1 \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  MAX_REF_BATCHES=50 \
  MAX_QUERIES_PER_HEAD=64 \
  MAX_REF_EXAMPLES=256 \
  QUERIES_PER_BATCH=all_tokens \
  STRIP_REF_SYSTEM_PROMPT=1 \
  NUM_BG_BATCHES="${NUM_BG_BATCHES:-64}" \
  SEED=42 \
  RUN_NAME="quality_p1_ropefix_armD" \
    "$PY_BIN" "$ROOT/examples/shared/am/initial_compaction.py"

  P01_CACHE="$(ls -1 "$P1_ROOT"/*/*/cache_last.pt | head -1)"
  echo "=== p01 cache: $P01_CACHE"
fi

# --- stages 2-5: continual writes + the 5x5 grid ------------------------------
exec "$PY_BIN" "$ROOT/examples/shared/am/continual_chain.py" \
  --dataset quality \
  --p01-cache "$P01_CACHE" \
  --tag "$TAG" \
  --gpu "$GPU" \
  "$@"
