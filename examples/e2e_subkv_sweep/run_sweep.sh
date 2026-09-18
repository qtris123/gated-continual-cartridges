#!/usr/bin/env bash
# ==============================================================================
# Full End-to-End Continual Compaction Sweep (Sub-KV Cache Sizes + Proportional top-t)
#
# Runs on a bare-metal GPU server (no Slurm required):
#   1. Initial Compaction (Phase 1 / p01) with sub-KV cache budget S
#   2. Multi-stage Continual AM Compaction (Phases 2-5 / p02-p05) with top-t
#   3. Evaluates 5x5 stage-by-eval matrix on held-out test sets after each stage
#
# Usage:
#   bash examples/e2e_subkv_sweep/run_sweep.sh [options]
#
# Examples:
#   # Run default sweep (512/32, 1024/64, 2048/128, 4096/256) on GPU 0 for QASPER:
#   bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --gpu 0
#
#   # Custom budgets and top-t:
#   bash examples/e2e_subkv_sweep/run_sweep.sh --dataset quality --budgets 512,1024 --top-ts 32,64 --gpu 1
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Defaults
DATASET="qasper"
GPU="${GPU:-0}"
EVAL_GPUS="${EVAL_GPUS:-$GPU}"
BUDGETS_STR="1024,2048,4096"
TOP_TS_STR="64,128,256"
FORCE=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --dataset <name>     Dataset name: qasper, quality, finqa, techqa (default: $DATASET)
  --gpu <id>           GPU index for compaction writes (default: $GPU)
  --eval-gpus <list>   GPU index or comma-separated list for evaluations (default: $EVAL_GPUS)
  --budgets <list>     Comma-separated sub-KV cache budgets (default: $BUDGETS_STR)
  --top-ts <list>      Comma-separated proportional top-t values (default: $TOP_TS_STR)
  --force              Rebuild and overwrite existing cache/eval artifacts
  --dry-run            Print execution plan without running
  -h, --help           Show this help message

Examples:
  # Run default sweep (1024/64, 2048/128, 4096/256) on GPU 0 for QASPER:
  bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --gpu 0

  # Include 512 (e.g. 512/32, 1024/64, 2048/128, 4096/256):
  bash examples/e2e_subkv_sweep/run_sweep.sh --budgets 512,1024,2048,4096 --top-ts 32,64,128,256
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)   DATASET="$2"; shift 2 ;;
    --gpu)       GPU="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --budgets)   BUDGETS_STR="$2"; shift 2 ;;
    --top-ts)    TOP_TS_STR="$2"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

# Resolve Python executable
if [[ -n "${CARTRIDGES_PYTHON:-}" ]]; then
  PY="$CARTRIDGES_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "$CONDA_PREFIX/bin/python" ]]; then
  PY="$CONDA_PREFIX/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CARTRIDGES_PYTHON="$PY"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$ROOT}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$ROOT/outputs}"

# Ensure data/qasper/phases symlink exists if data/phases/qasper exists
if [[ "$DATASET" == "qasper" && ! -d "$ROOT/data/qasper/phases" && -d "$ROOT/data/phases/qasper" ]]; then
  echo "Ensuring data/qasper/phases -> ../phases/qasper symlink"
  mkdir -p "$ROOT/data/qasper"
  ln -sfn ../phases/qasper "$ROOT/data/qasper/phases"
fi

IFS=',' read -r -a BUDGETS <<< "$BUDGETS_STR"
IFS=',' read -r -a TOP_TS <<< "$TOP_TS_STR"

if [[ "${#BUDGETS[@]}" -ne "${#TOP_TS[@]}" ]]; then
  echo "Error: Number of budgets (${#BUDGETS[@]}) must match number of top_ts (${#TOP_TS[@]})." >&2
  exit 1
fi

LOGDIR="$ROOT/logs/e2e_subkv_sweep"
RECIPEDIR="$ROOT/outputs/recipes/subkv_sweep"
mkdir -p "$LOGDIR" "$RECIPEDIR"

# Preflight data check
MISSING_DATA=0
if ! "$PY" -c "
import sys
from examples.shared.am.continual_env import spec
s = spec('$DATASET')
for p in range(1, 6):
    ep = s.eval_path(p)
    if not ep.exists():
        print(f'Missing eval file: {ep}', file=sys.stderr)
        sys.exit(1)
for p in range(1, 6):
    sp = s.synth_path(p)
    if not sp.exists():
        print(f'Missing synth data file: {sp}', file=sys.stderr)
        sys.exit(1)
" 2>&1; then
  MISSING_DATA=1
fi

if (( MISSING_DATA )); then
  echo ""
  echo "======================================================================"
  echo " Notice: Dataset files for '$DATASET' are not fully hydrated on this host."
  echo " To download all required training/eval data from Hugging Face, run:"
  echo "     bash scripts/prepare_artifacts.sh --which data"
  echo "======================================================================"
  echo ""
  if (( ! DRY_RUN )); then
    echo "Aborting run due to missing data. Run with --dry-run to preview plan." >&2
    exit 1
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"

echo "======================================================================"
echo " Starting E2E Continual Compaction Sweep"
echo " Dataset:   $DATASET"
echo " GPU:       $GPU (Evals: $EVAL_GPUS)"
echo " Python:    $PY"
echo " Budgets:   ${BUDGETS[*]}"
echo " Top-Ts:    ${TOP_TS[*]}"
echo " Root:      $ROOT"
echo "======================================================================"

for i in "${!BUDGETS[@]}"; do
  SIZE="${BUDGETS[$i]}"
  TOP_T="${TOP_TS[$i]}"
  TAG="e2e_budget${SIZE}_topt${TOP_T}"
  RECIPE="$RECIPEDIR/${TAG}.yaml"
  P1_ROOT="$ROOT/outputs/experiments/subkv_sweep_${DATASET}/${TAG}/p01"
  SWEEP_LOG="$LOGDIR/${DATASET}_${TAG}_${TS}.log"

  echo ""
  echo ">>> [Arm $((i+1))/${#BUDGETS[@]}] Budget=$SIZE, top_t=$TOP_T (Tag: $TAG) <<<"
  echo "    Log file: $SWEEP_LOG"

  # 1. Generate size-specific recipe YAML matching canonical experiments (e.g. fullkv_topt512.yaml)
  cat <<EOF > "$RECIPE"
# Auto-generated for sub-KV cache sweep (Budget=$SIZE, top_t=$TOP_T)
# Matches fullkv_topt512.yaml and topt*.yaml:
# - Attention-output matching: delta_weight=0.01, ridge_lambda=0.0
# - Highest-attention keys: key_mode=highest_attention, key_reposition=true
# - No attention-bias: beta.enabled=false
# - Fixed RoPE repositioning: rope_theta=5000000.0
p01:
  num_tokens: $SIZE
  key_select: highest_attention
  granularity: per_head
  enable_beta: 0
  rebake_key_positions: 1
  rope_theta: model
  global_teacher_positions: 1
  ridge_lambda: 1e-4
  ridge_scale: spectral
  max_queries_per_head: 64
  max_ref_batches: 50
  max_ref_examples: 256
  queries_per_batch: all_tokens
  strip_ref_system_prompt: 1
  num_bg_batches: 64
  seed: 42

slots:
  top_t: $TOP_T
  granularity: per_layer
  slot_selection: tfidf
  use_idf: false
  safe_fraction: 1.0
  safe_metric: redundancy
  usage_penalty_lambda: 0.0
  min_top_t_per_layer: 1
  background_top_k_per_batch: 128
  idf_prior_weight: 0.0
  idf_smoothing: 1.0
  mass_redundancy_alpha: 0.5
  num_background_batches: 999999999
  redundancy_ridge_rel: 1.0e-06

keys:
  key_mode: highest_attention
  key_reposition: true

beta:
  enabled: false
  fit_scope: selected
  beta_box: 3.0
  nnls_driver: gelsd
  nnls_iters: 2
  target_mode: residual

queries:
  max_queries_per_head: 64
  max_ref_examples_per_doc: 32
  onpolicy_layers: 0
  onpolicy_refresh_doc_kv: false
  queries_per_batch: all_tokens
  ref_batch_limit: 5
  seed_offset: 0

objective:
  delta_weight: 0.01
  ridge_lambda: 0.0
  ridge_lambda_min: 0.0
  ridge_scale: spectral
  enable_old_reference_guard: false
  old_ref_data_path: null
  old_ref_max_examples: 64
  old_reference_weight: 0.0
  oracle_write: false
  oracle_write_assign: mass_ranked

rope_theta: 5000000.0
EOF

  if (( DRY_RUN )); then
    echo "    [dry-run] Wrote recipe to $RECIPE"
    echo "    [dry-run] Would build Phase 1 cache under $P1_ROOT"
    echo "    [dry-run] Would run continual chain p02-p05 with tag $TAG"
    continue
  fi

  # 2. Step 1: Initial Compaction (Phase 1)
  echo "--- Step 1: Initial Compaction (p01, $SIZE slots) ---"
  BUILD_ARGS=(
    --dataset "$DATASET"
    --recipe-config "$RECIPE"
    --p1-root "$P1_ROOT"
    --gpu "$GPU"
  )
  if (( FORCE )); then
    BUILD_ARGS+=(--force)
  fi

  "$PY" "$ROOT/examples/shared/am/build_p01.py" "${BUILD_ARGS[@]}" 2>&1 | tee -a "$SWEEP_LOG"

  # Find generated or existing p01 cache
  P01_CACHE="$(find "$P1_ROOT" -name "cache_last.pt" 2>/dev/null | head -1 || true)"
  if [[ -z "$P01_CACHE" || ! -f "$P01_CACHE" ]]; then
    echo "Error: Failed to find p01 cache under $P1_ROOT" >&2
    exit 1
  fi
  echo "    P01 Cache ready: $P01_CACHE"

  # 3. Step 2: Continual Compaction (Phases 2 to 5)
  echo "--- Step 2: Continual Compaction (Stages p02-p05, top_t=$TOP_T) ---"
  CHAIN_ARGS=(
    --dataset "$DATASET"
    --method am
    --p01-cache "$P01_CACHE"
    --tag "$TAG"
    --recipe-config "$RECIPE"
    --gpu "$GPU"
    --eval-gpus "$EVAL_GPUS"
  )

  "$PY" "$ROOT/examples/shared/am/run_chain.py" "${CHAIN_ARGS[@]}" 2>&1 | tee -a "$SWEEP_LOG"

  MAT_JSON="$ROOT/outputs/evaluations/$DATASET/$TAG/teacher-forced-logppl-v1/matrix.json"
  if [[ -f "$MAT_JSON" ]]; then
    echo "    Successfully generated matrix: $MAT_JSON"
  else
    echo "    Warning: matrix.json not found at $MAT_JSON (check log)"
  fi
done

echo ""
echo "======================================================================"
echo " All arms completed successfully at $(date)!"
echo " Results available in: outputs/evaluations/$DATASET/"
echo " Logs stored in:        $LOGDIR/"
echo "======================================================================"
