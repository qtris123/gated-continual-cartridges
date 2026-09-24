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

# One chain per GPU. Leave the other cores free so seven chains do not each
# pin every CPU and starve the GPU launch threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Defaults
DATASET_DEFAULT="qasper"
DATASET=""
DATASETS_STR=""
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
GPU="${GPU:-0}"
EVAL_GPUS="${EVAL_GPUS:-$GPU}"
BUDGET_ARG=""
BUDGETS_STR="1024,2048,4096"
TOP_TS_STR="64,128,256"
MAX_QUERIES_ARG=""
DELTA_WEIGHT_ARG=""
TAG_SUFFIX=""
FORCE=0
DRY_RUN=0
EVAL_ACCURACY=0

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --model <id>         HuggingFace model ID (default: $MODEL_NAME)
  --datasets <list>    Comma-separated datasets: qasper, quality, finqa, techqa, longhealth
  --dataset <name>     Single dataset: qasper, quality, finqa, techqa, longhealth (default: $DATASET_DEFAULT)
  --gpu <id>           GPU index for compaction writes (default: $GPU)
  --eval-gpus <list>   GPU index or comma-separated list for evaluations (default: $EVAL_GPUS)
  --budget <size>      Single sub-KV cache budget to sweep across multiple top-ts (e.g. 16384)
  --budgets <list>     Comma-separated sub-KV cache budgets (default: $BUDGETS_STR)
  --top-ts <list>      Comma-separated proportional top-t values (default: $TOP_TS_STR)
  --max-queries-per-head <N> Max queries per head (default: 1024)
  --delta-weight <val> Explicit delta_weight (default: linearly scaled with max queries: 0.16 for 1024)
  --tag-suffix <str>   Suffix to append to lineage tag and p01 root (e.g. _q64, _q1024)
  --eval-accuracy      Run additional 5x5 text generation accuracy evaluation (for QuALITY/LongHealth MCQ)
  --force              Rebuild and overwrite existing cache/eval artifacts
  --dry-run            Print execution plan without running
  -h, --help           Show this help message

Examples:
  # Run default sweep (1024/64, 2048/128, 4096/256) on GPU 0 for QASPER (Qwen):
  bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --gpu 0

  # Run on Llama 3.2 3B:
  bash examples/e2e_subkv_sweep/run_sweep.sh --model meta-llama/Llama-3.2-3B-Instruct --dataset qasper --gpu 0

  # Sweep top-ts for a single budget (e.g. 16384 slots) on QuALITY with MCQ accuracy:
  bash examples/e2e_subkv_sweep/run_sweep.sh --budget 16384 --top-ts 1024,4096,8192 --dataset quality --eval-accuracy --gpu 0

  # Run across multiple datasets:
  bash examples/e2e_subkv_sweep/run_sweep.sh --datasets qasper,quality --gpu 0
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL_NAME="$2"; shift 2 ;;
    --datasets)  DATASETS_STR="$2"; shift 2 ;;
    --dataset)   DATASET="$2"; shift 2 ;;
    --gpu)       GPU="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --budget)    BUDGET_ARG="$2"; shift 2 ;;
    --budgets)   BUDGETS_STR="$2"; shift 2 ;;
    --top-ts)    TOP_TS_STR="$2"; shift 2 ;;
    --top-t)     TOP_TS_STR="$2"; shift 2 ;;
    --max-queries-per-head) MAX_QUERIES_ARG="$2"; shift 2 ;;
    --delta-weight) DELTA_WEIGHT_ARG="$2"; shift 2 ;;
    --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
    --eval-accuracy) EVAL_ACCURACY=1; shift ;;
    --force)     FORCE=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

if [[ -n "$DATASETS_STR" ]]; then
  IFS=',' read -r -a DATASETS <<< "$DATASETS_STR"
elif [[ -n "$DATASET" ]]; then
  DATASETS=("$DATASET")
else
  DATASETS=("$DATASET_DEFAULT")
fi

IFS=',' read -r -a TOP_TS <<< "$TOP_TS_STR"

if [[ -n "$BUDGET_ARG" ]]; then
  BUDGETS=()
  for _ in "${TOP_TS[@]}"; do
    BUDGETS+=("$BUDGET_ARG")
  done
else
  IFS=',' read -r -a BUDGETS <<< "$BUDGETS_STR"
  if [[ "${#BUDGETS[@]}" -eq 1 && "${#TOP_TS[@]}" -gt 1 ]]; then
    SINGLE_BUDGET="${BUDGETS[0]}"
    BUDGETS=()
    for _ in "${TOP_TS[@]}"; do
      BUDGETS+=("$SINGLE_BUDGET")
    done
  fi
fi

if [[ "${#BUDGETS[@]}" -ne "${#TOP_TS[@]}" ]]; then
  echo "Error: Number of budgets (${#BUDGETS[@]}) must match number of top_ts (${#TOP_TS[@]})." >&2
  exit 1
fi
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

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CARTRIDGES_PYTHON="$PY"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$ROOT}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$ROOT/outputs}"
export MODEL_NAME="$MODEL_NAME"

if [[ "$MODEL_NAME" =~ [Ll]lama ]]; then
  MODEL_SLUG="llama3_2_3b"
  ROPE_THETA_VAL="model"
elif [[ "$MODEL_NAME" =~ [Qq]wen ]]; then
  MODEL_SLUG="qwen3_4b"
  ROPE_THETA_VAL="5000000.0"
else
  MODEL_SLUG="$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
  ROPE_THETA_VAL="model"
fi

LOGDIR="$ROOT/logs/e2e_subkv_sweep"
RECIPEDIR="$ROOT/outputs/recipes/subkv_sweep"
mkdir -p "$LOGDIR" "$RECIPEDIR"

for DATASET in "${DATASETS[@]}"; do
  # Ensure data/qasper/phases symlink exists if data/phases/qasper exists
  if [[ "$DATASET" == "qasper" && ! -d "$ROOT/data/qasper/phases" && -d "$ROOT/data/phases/qasper" ]]; then
    echo "Ensuring data/qasper/phases -> ../phases/qasper symlink"
    mkdir -p "$ROOT/data/qasper"
    ln -sfn ../phases/qasper "$ROOT/data/qasper/phases"
  fi

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
      echo "Aborting run due to missing data for '$DATASET'. Run with --dry-run to preview plan." >&2
      exit 1
    fi
  fi

  TS="$(date +%Y%m%d_%H%M%S)"

  echo "======================================================================"
  echo " Starting E2E Continual Compaction Sweep"
  echo " Model:     $MODEL_NAME (Slug: $MODEL_SLUG)"
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
    TAG="${MODEL_SLUG}_budget${SIZE}_topt${TOP_T}${TAG_SUFFIX}"
    P1_PREFIX="${MODEL_SLUG}_"
    RECIPE="$RECIPEDIR/${TAG}.yaml"
    P1_ROOT="$ROOT/outputs/experiments/subkv_sweep_${DATASET}/${P1_PREFIX}budget${SIZE}${TAG_SUFFIX}/p01"
    SWEEP_LOG="$LOGDIR/${DATASET}_${TAG}_${TS}.log"

  if [[ -n "$MAX_QUERIES_ARG" ]]; then
    MAX_QUERIES_PER_HEAD="$MAX_QUERIES_ARG"
  else
    MAX_QUERIES_PER_HEAD="1024"
  fi

  if [[ -n "$DELTA_WEIGHT_ARG" ]]; then
    DELTA_WEIGHT="$DELTA_WEIGHT_ARG"
  else
    # Linearly scale delta_weight: lambda_Delta = 0.01 * (N_q / 64) -> 0.16 for 1024 queries
    DELTA_WEIGHT=$(awk -v q="$MAX_QUERIES_PER_HEAD" 'BEGIN { v = 0.01 * (q / 64); printf "%g\n", v }')
  fi

  echo ""
  echo ">>> [Arm $((i+1))/${#BUDGETS[@]}] Budget=$SIZE, top_t=$TOP_T, max_queries_per_head=$MAX_QUERIES_PER_HEAD, delta_weight=$DELTA_WEIGHT (Tag: $TAG) <<<"
  echo "    Log file: $SWEEP_LOG"

  # 1. Generate size-specific recipe YAML matching canonical experiments (e.g. fullkv_topt512.yaml)
  cat <<EOF > "$RECIPE"
# Auto-generated for sub-KV cache sweep (Model=$MODEL_NAME, Budget=$SIZE, top_t=$TOP_T, max_queries_per_head=$MAX_QUERIES_PER_HEAD)
# - Stage 1 (p01): Arm-D compaction with spectral ridge_lambda=1e-4
# - Stages 2-5 (p02-p05): Continual delta matching with delta_weight=$DELTA_WEIGHT, ridge_lambda=0.0
# - Highest-attention keys: key_mode=highest_attention, key_reposition=true
# - No attention-bias: beta.enabled=false
# - Fixed RoPE repositioning: rope_theta=$ROPE_THETA_VAL
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
  max_queries_per_head: $MAX_QUERIES_PER_HEAD
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
  max_queries_per_head: $MAX_QUERIES_PER_HEAD
  max_ref_examples_per_doc: 32
  onpolicy_layers: 0
  onpolicy_refresh_doc_kv: false
  queries_per_batch: all_tokens
  ref_batch_limit: 5
  seed_offset: 0

objective:
  delta_weight: $DELTA_WEIGHT
  ridge_lambda: 0.0
  ridge_lambda_min: 0.0
  ridge_scale: spectral
  enable_old_reference_guard: false
  old_ref_data_path: null
  old_ref_max_examples: 64
  old_reference_weight: 0.0
  oracle_write: false
  oracle_write_assign: mass_ranked

rope_theta: $ROPE_THETA_VAL
EOF

  if (( DRY_RUN )); then
    echo "    [dry-run] Wrote recipe to $RECIPE"
    echo "    [dry-run] Would build Phase 1 cache under $P1_ROOT"
    echo "    [dry-run] Would run continual chain p02-p05 with tag $TAG"
    if (( EVAL_ACCURACY )) || [[ "$DATASET" == "longhealth" ]]; then
      echo "    [dry-run] Would run 5x5 generation accuracy evaluation for tag $TAG"
    fi
    echo "    [dry-run] Would record slot frequency artifacts under outputs/evaluations/$DATASET/$TAG/slot_frequency"
    continue
  fi

    # 2. Step 1: Initial Compaction (Phase 1)
    P01_CACHE="$(find "$P1_ROOT" -name "cache_last.pt" 2>/dev/null | head -1 || true)"
    if [[ -z "$P01_CACHE" ]]; then
      LEGACY_P1="$ROOT/outputs/experiments/subkv_sweep_${DATASET}/${TAG}/p01"
      P01_CACHE="$(find "$LEGACY_P1" -name "cache_last.pt" 2>/dev/null | head -1 || true)"
      if [[ -n "$P01_CACHE" ]]; then
        P1_ROOT="$LEGACY_P1"
      fi
    fi

    if [[ -n "$P01_CACHE" && -f "$P01_CACHE" && $FORCE -eq 0 ]]; then
      echo ">>> Found existing Phase 1 cache for $DATASET ($SIZE slots): $P01_CACHE (skipping build) <<<"
    else
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

      P01_CACHE="$(find "$P1_ROOT" -name "cache_last.pt" 2>/dev/null | head -1 || true)"
      if [[ -z "$P01_CACHE" || ! -f "$P01_CACHE" ]]; then
        echo "Error: Failed to find p01 cache under $P1_ROOT" >&2
        exit 1
      fi
      echo "    P01 Cache ready: $P01_CACHE"
    fi

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
      echo "    Successfully generated perplexity matrix: $MAT_JSON"
    else
      echo "    Warning: matrix.json not found at $MAT_JSON (check log)"
    fi

    # 4. Step 3: Record Slot Frequency and Geometry Artifacts
    echo "--- Step 3: Record Slot Frequency Artifacts ($DATASET, $TAG) ---"
    SLOT_FREQ_DIR="$ROOT/outputs/evaluations/$DATASET/$TAG/slot_frequency"
    "$PY" -m examples.shared.am.record_slot_frequency \
      --dataset "$DATASET" \
      --tag "$TAG" \
      --output-dir "$SLOT_FREQ_DIR" 2>&1 | tee -a "$SWEEP_LOG" || {
        echo "    Notice: Slot frequency recording produced warnings or was skipped (check log)"
      }

    # 5. Optional Step 4: Generation Accuracy Matrix (for QuALITY/LongHealth MCQ)
    if (( EVAL_ACCURACY )) || [[ "$DATASET" == "longhealth" ]] || [[ "$DATASET" == "quality" ]]; then
      echo "--- Step 4: Generation Accuracy Matrix ($DATASET, $TAG) ---"
      ACC_LOG="$LOGDIR/acc_${DATASET}_${TAG}_${TS}.log"
      bash "$ROOT/examples/shared/evaluate/run_accuracy.sh" "$DATASET" "$TAG" "$EVAL_GPUS" 2>&1 | tee -a "$ACC_LOG"
      ACC_JSON="$ROOT/outputs/evaluations/$DATASET/$TAG/accuracy-freeform-mc-options-primeAnswer-v1/matrix.json"
      if [[ -f "$ACC_JSON" ]]; then
        echo "    Successfully generated accuracy matrix: $ACC_JSON"
      fi
    fi
  done
done

echo ""
echo "======================================================================"
echo " All datasets and arms completed successfully at $(date)!"
echo " Results available in: outputs/evaluations/{${DATASETS[*]}}/"
echo " Logs stored in:        $LOGDIR/"
echo "======================================================================"
