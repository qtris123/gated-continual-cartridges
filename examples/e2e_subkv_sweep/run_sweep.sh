#!/usr/bin/env bash
# ==============================================================================
# Continual AM Overlap Sweep (Encouraging Slot Reuse Across Stages)
#
# Runs on a bare-metal GPU server (no Slurm required):
#   1. Initial Compaction (Phase 1 / p01) with sub-KV cache budget S (built once per dataset)
#   2. Multi-stage Continual AM Compaction (Phases 2-5 / p02-p05) sweeping over lambda:
#      - Exponent reversed in SlotSelector: score * (1 + usage)^lambda
#      - lambda = 0.0: Baseline (no overlap bonus, bit-identical to standard)
#      - lambda > 0.0: Encourages reusing previously written slots across stages
#   3. Evaluates 5x5 stage-by-eval matrix on held-out test sets after each stage
#
# Usage:
#   bash examples/e2e_subkv_sweep/run_sweep.sh [options]
#
# Examples:
#   # Run default sweep (lambda: 0, 0.25, 0.5, 0.75, 1.0, 2.0) on Quality and QASPER on GPU 0:
#   bash examples/e2e_subkv_sweep/run_sweep.sh --gpu 0
#
#   # Run on single dataset (e.g. Quality):
#   bash examples/e2e_subkv_sweep/run_sweep.sh --dataset quality --gpu 0
#
#   # Custom lambdas or budgets:
#   bash examples/e2e_subkv_sweep/run_sweep.sh --datasets quality,qasper --lambdas 0.0,0.5,1.0 --budget 1024 --top-t 64 --gpu 1
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Defaults
DATASETS_STR="quality,qasper"
LAMBDAS_STR="0.0,0.25,0.5,0.75,1.0,2.0"
BUDGET=1024
TOP_T=64
GPU="${GPU:-0}"
EVAL_GPUS="${EVAL_GPUS:-$GPU}"
FORCE=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --datasets <list>    Comma-separated datasets: quality, qasper (default: $DATASETS_STR)
  --dataset <name>     Single dataset alias: quality or qasper
  --lambdas <list>     Comma-separated lambda values for overlap bonus (default: $LAMBDAS_STR)
  --budget <size>      Sub-KV cache budget for Phase 1 (default: $BUDGET)
  --top-t <int>        Proportional top-t updates per layer (default: $TOP_T)
  --gpu <id>           GPU index for compaction writes (default: $GPU)
  --eval-gpus <list>   GPU index or comma-separated list for evaluations (default: $EVAL_GPUS)
  --force              Rebuild and overwrite existing cache/eval artifacts
  --dry-run            Print execution plan without running
  -h, --help           Show this help message

Examples:
  # Run overlap sweep on Quality and QASPER (default lambdas: 0, 0.25, 0.5, 0.75, 1.0, 2.0):
  bash examples/e2e_subkv_sweep/run_sweep.sh --gpu 0

  # Run on Quality only:
  bash examples/e2e_subkv_sweep/run_sweep.sh --dataset quality --gpu 0

  # Run on QASPER only:
  bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --gpu 0
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|--datasets) DATASETS_STR="$2"; shift 2 ;;
    --lambdas)            LAMBDAS_STR="$2"; shift 2 ;;
    --budget)             BUDGET="$2"; shift 2 ;;
    --top-t)              TOP_T="$2"; shift 2 ;;
    --gpu)                GPU="$2"; shift 2 ;;
    --eval-gpus)          EVAL_GPUS="$2"; shift 2 ;;
    --force)              FORCE=1; shift ;;
    --dry-run)            DRY_RUN=1; shift ;;
    -h|--help)            usage ;;
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

IFS=',' read -r -a DATASETS <<< "$DATASETS_STR"
IFS=',' read -r -a LAMBDAS <<< "$LAMBDAS_STR"

num_slug() {
  local val="$1"
  if [[ "$val" =~ ^0*([0-9]+)\.0+$ ]] || [[ "$val" == "0.0" ]] || [[ "$val" == "0" ]]; then
    echo "${BASH_REMATCH[1]:-0}"
  elif [[ "$val" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}p${BASH_REMATCH[2]}"
  else
    echo "${val//./p}"
  fi
}

LOGDIR="$ROOT/logs/overlap_sweep"
RECIPEDIR="$ROOT/outputs/recipes/overlap_sweep"
mkdir -p "$LOGDIR" "$RECIPEDIR"

TS="$(date +%Y%m%d_%H%M%S)"

echo "======================================================================"
echo " Starting Continual AM Overlap Sweep"
echo " Datasets:  ${DATASETS[*]}"
echo " Lambdas:   ${LAMBDAS[*]}"
echo " Budget:    $BUDGET (top_t: $TOP_T)"
echo " GPU:       $GPU (Evals: $EVAL_GPUS)"
echo " Python:    $PY"
echo " Root:      $ROOT"
echo "======================================================================"

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "######################################################################"
  echo " Processing Dataset: $DATASET"
  echo "######################################################################"

  # Ensure data/qasper/phases symlink exists if data/phases/qasper exists
  if [[ "$DATASET" == "qasper" && ! -d "$ROOT/data/qasper/phases" && -d "$ROOT/data/phases/qasper" ]]; then
    echo "Ensuring data/qasper/phases -> ../phases/qasper symlink"
    mkdir -p "$ROOT/data/qasper"
    ln -sfn ../phases/qasper "$ROOT/data/qasper/phases"
  fi

  # Preflight data check for this dataset
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

  # Step 1: Initial Compaction (Phase 1 / p01)
  # Phase 1 is independent of lambda; build it ONCE per dataset and budget.
  P1_ROOT="$ROOT/outputs/experiments/overlap_sweep_${DATASET}/budget${BUDGET}/p01"
  P1_RECIPE="$RECIPEDIR/p01_budget${BUDGET}.yaml"
  P1_LOG="$LOGDIR/${DATASET}_p01_budget${BUDGET}_${TS}.log"

  cat <<EOF > "$P1_RECIPE"
# Auto-generated Phase 1 initial compaction recipe (Budget=$BUDGET)
p01:
  num_tokens: $BUDGET
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
EOF

  P01_CACHE="$(find "$P1_ROOT" -name "cache_last.pt" 2>/dev/null | head -1 || true)"
  if [[ -n "$P01_CACHE" && -f "$P01_CACHE" && $FORCE -eq 0 ]]; then
    echo ">>> Found existing Phase 1 cache for $DATASET: $P01_CACHE (skipping build) <<<"
  else
    echo ">>> Step 1: Building Phase 1 Initial Compaction for $DATASET ($BUDGET slots) <<<"
    if (( DRY_RUN )); then
      echo "    [dry-run] Would build Phase 1 cache under $P1_ROOT"
      P01_CACHE="$P1_ROOT/dummy/cache_last.pt"
    else
      BUILD_ARGS=(
        --dataset "$DATASET"
        --recipe-config "$P1_RECIPE"
        --p1-root "$P1_ROOT"
        --gpu "$GPU"
      )
      if (( FORCE )); then
        BUILD_ARGS+=(--force)
      fi
      "$PY" "$ROOT/examples/shared/am/build_p01.py" "${BUILD_ARGS[@]}" 2>&1 | tee -a "$P1_LOG"
      P01_CACHE="$(find "$P1_ROOT" -name "cache_last.pt" 2>/dev/null | head -1 || true)"
      if [[ -z "$P01_CACHE" || ! -f "$P01_CACHE" ]]; then
        echo "Error: Failed to find p01 cache under $P1_ROOT" >&2
        exit 1
      fi
      echo "    Phase 1 Cache ready: $P01_CACHE"
    fi
  fi

  # Step 2: Multi-stage Continual AM Compaction (Phases 2-5) sweeping over lambdas
  for idx in "${!LAMBDAS[@]}"; do
    LAM="${LAMBDAS[$idx]}"
    LAM_SLUG="$(num_slug "$LAM")"
    TAG="overlap_lam${LAM_SLUG}"
    RECIPE="$RECIPEDIR/${TAG}.yaml"
    SWEEP_LOG="$LOGDIR/${DATASET}_${TAG}_${TS}.log"

    echo ""
    echo ">>> [$DATASET Arm $((idx+1))/${#LAMBDAS[@]}] lambda=$LAM (Tag: $TAG) <<<"
    echo "    Log file: $SWEEP_LOG"

    cat <<EOF > "$RECIPE"
# Auto-generated for overlap sweep (Dataset=$DATASET, Budget=$BUDGET, top_t=$TOP_T, lambda=$LAM)
# - Stage 1 (p01): Arm-D compaction with spectral ridge_lambda=1e-4
# - Stages 2-5 (p02-p05): Continual delta matching with delta_weight=0.01, ridge_lambda=0.0
# - Overlap bonus: usage_penalty_lambda=$LAM (reversed exponent (1+u)^lambda encourages slot reuse)
# - Highest-attention keys: key_mode=highest_attention, key_reposition=true
# - No attention-bias: beta.enabled=false
# - Fixed RoPE repositioning: rope_theta=5000000.0
p01:
  num_tokens: $BUDGET
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
  usage_penalty_lambda: $LAM
  usage_decay: 1.0
  usage_penalty_mode: mult
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
      echo "    [dry-run] Would run continual chain p02-p05 with tag $TAG using p01-cache $P01_CACHE"
      continue
    fi

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
    MAT_CSV="$ROOT/outputs/evaluations/$DATASET/$TAG/teacher-forced-logppl-v1/matrix.csv"
    if [[ -f "$MAT_CSV" ]]; then
      echo "    Successfully generated matrix: $MAT_CSV"
    elif [[ -f "$MAT_JSON" ]]; then
      echo "    Successfully generated matrix: $MAT_JSON"
    else
      echo "    Warning: matrix file not found at $MAT_JSON (check log: $SWEEP_LOG)"
    fi
  done
done

echo ""
echo "======================================================================"
echo " All datasets and lambda arms completed successfully at $(date)!"
echo " Results available in: outputs/evaluations/{quality,qasper}/overlap_lam*/"
echo " Logs stored in:        $LOGDIR/"
echo "======================================================================"
