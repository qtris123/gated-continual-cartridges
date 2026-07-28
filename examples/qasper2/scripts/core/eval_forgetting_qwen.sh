# Forgetting Eval (Qwen3-4B): Phase 1 retention + Phase 2 acquisition for qasper.
#
# For each Phase 2 checkpoint listed in EXPERIMENTS, this script runs
# train/eval_forgetting.py twice:
#   1. Against the Phase 1 eval parquet (QA task)  → measures forgetting
#   2. Against the Phase 2 eval parquet (MT task)  → measures acquisition
#
# Jobs are dispatched round-robin across NUM_GPUS GPUs so all GPUs run in
# parallel (up to NUM_GPUS concurrent evals at a time). With 5 checkpoints x
# 2 eval sets = 10 jobs and NUM_GPUS=4, the first 4 jobs launch immediately
# (one per GPU); jobs 5–10 launch as each GPU slot frees up.
#
# Results (perplexity) are logged to wandb and written to
#   $CARTRIDGES_OUTPUT_DIR/qasper_forgetting_eval_qwen_per-layer_<pid>/
#
# Labels intentionally omit "num-tokens-512" — every run in this sweep shares
# the same num-tokens budget, so downstream figures group by top-t only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-4}"

# All wandb runs from this batch land under one group so the analysis
# pipeline can pull them with a single (entity, project, group) filter.
WANDB_GROUP="${WANDB_GROUP:-qwen-qasper-per-layer-num-tokens-512 - [granularity x forgetting/learning]}"

# Phase 1 task: QA — used to measure forgetting.
# Phase 2 task: MT — used to measure acquisition.
QA_EVAL="${QA_EVAL:-${CARTRIDGES_DIR}/examples/qasper2/qasper_eval_QA.parquet}"
MT_EVAL="${MT_EVAL:-${CARTRIDGES_DIR}/examples/qasper2/qasper_eval_MT.parquet}"

CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"
RESULTS_DIR="${CARTRIDGES_OUTPUT_DIR}/qasper_forgetting_eval_qwen_per-layer_num-tokens-512_${SLURM_JOB_ID:-$$}"
mkdir -p "${RESULTS_DIR}"

# =============================================================================
# Experiment definitions
# Format: "CHECKPOINT_PATH | RUN_LABEL"
#
# Qwen3-4B per-layer sweep at num-tokens=512, latest checkpoint cache-step625.pt.
# Labels intentionally drop the shared "num-tokens-512" tag — figures group by top-t.
# =============================================================================
EXPERIMENTS=(
  "${CARTRIDGES_OUTPUT_DIR}/2026-06-22-08-31-41-initial_sparse_qwen_qasper_per-layer_all-reduce_num-tokens-512/ef9e7f9d-7b66-4369-86d6-bdecd9db9efd/cache-step622.pt | qwen-per-layer-baseline"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-22-10-01-52-continual_sparse_qwen_qasper_per-layer_top-t-32_all_reduce_num-tokens-512/e79d2482-c57c-4067-b854-06eb8fcf98b5/cache-step625.pt | qwen-per-layer-top-32"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-22-10-01-54-continual_sparse_qwen_qasper_per-layer_top-t-64_all-reduce_num-tokens-512/e475a985-f0b2-4056-bf7f-10332afa9187/cache-step625.pt | qwen-per-layer-top-64"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-22-14-17-15-continual_sparse_qwen_qasper_per-layer_top-t-128_all-reduce_num-tokens-512/3558d30b-f925-48f8-a3bb-39a27247187d/cache-step625.pt | qwen-per-layer-top-128"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-22-11-47-27-continual_sparse_qwen_qasper_per-layer_top-t-256_all-reduce_num-tokens-512/52e06c16-07ca-491a-9d39-2dfa56fc00dc/cache-step625.pt | qwen-per-layer-top-256"
)

echo "=========================================="
echo "Qasper Forgetting Eval (Qwen3-4B, per-layer)"
echo "Phase 1 retention (QA) + Phase 2 acquisition (MT)"
echo "=========================================="
echo "Host=$(hostname)"
echo "Started at: $(date)"
echo "Model:         ${MODEL_NAME}"
echo "Results dir:   ${RESULTS_DIR}"
echo "Experiments:   ${#EXPERIMENTS[@]} checkpoints × 2 eval sets"
echo "Parallel GPUs: ${NUM_GPUS}"
echo "WandB group:   ${WANDB_GROUP:-<unset>}"
echo ""

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
  echo "WARNING: No experiments defined — edit the EXPERIMENTS array in this script"
  exit 0
fi

# Load modules only when running under a module system (e.g. SLURM cluster)
if command -v module >/dev/null 2>&1; then
  module load gcc/11.4.1 2>/dev/null || true
  module load cuda/12.1.0 2>/dev/null || true
fi

echo "CUDA_HOME=${CUDA_HOME:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  echo "(nvidia-smi not found)"
fi

# Activate the Python environment where pydrantic + cartridges are installed.
# Prefer a repo-local .venv; otherwise fall back to a conda env (default: cartridges).
# Mirrors train_continual_sparse.sh so the same host setup works for eval too.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cartridges}"
if [ -f "$CARTRIDGES_DIR/.venv/bin/activate" ]; then
  echo "Activating .venv..."
  source "$CARTRIDGES_DIR/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Activating conda env '$CONDA_ENV_NAME'..."
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
else
  echo "Error: no .venv at $CARTRIDGES_DIR/.venv and conda env '$CONDA_ENV_NAME' not found." >&2
  echo "       Set CONDA_ENV_NAME to override the conda env name." >&2
  exit 1
fi
echo "Python: $(which python3)"
echo ""

# =============================================================================
# Main eval loop — parallel dispatch across NUM_GPUS GPUs
#
# Jobs are dispatched round-robin: job 0 → GPU 0, job 1 → GPU 1, …
# Before dispatching job N, we wait for job N-NUM_GPUS to finish, so at most
# NUM_GPUS evals run concurrently. Logs for each eval go to separate files so
# interleaved output doesn't get mixed.
# =============================================================================
TOTAL=$(( ${#EXPERIMENTS[@]} * 2 ))
JOB_IDX=0
PIDS=()     # background PIDs in dispatch order

run_eval() {
  local gpu="$1"
  local run_name="$2"
  local ckpt_path="$3"
  local eval_data="$4"
  local out_dir="$5"
  local disp_idx="$6"

  mkdir -p "$out_dir"
  echo "══════════════════════════════════════════"
  echo "  [${disp_idx}/${TOTAL}] ${run_name}  →  GPU ${gpu}"
  echo "  Checkpoint: ${ckpt_path}"
  echo "  Log: ${out_dir}/eval.log"
  echo "══════════════════════════════════════════"

  CUDA_VISIBLE_DEVICES="$gpu" \
  CHECKPOINT_PATH="$ckpt_path" \
  EVAL_DATA_PATH="$eval_data" \
  MODEL_NAME="$MODEL_NAME" \
  RUN_NAME="$run_name" \
  BATCH_SIZE="$BATCH_SIZE" \
  WANDB_GROUP="$WANDB_GROUP" \
  python3 "${CARTRIDGES_DIR}/examples/qasper2/train/eval_forgetting.py" \
    >"${out_dir}/eval.log" 2>&1
  echo "  Done: ${run_name}  (GPU ${gpu})"
}

for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r CKPT_PATH LABEL <<< "$ENTRY"
  CKPT_PATH=$(echo "$CKPT_PATH" | xargs)
  LABEL=$(echo "$LABEL" | xargs)

  if [ ! -f "$CKPT_PATH" ]; then
    echo "WARNING: checkpoint not found, skipping: ${CKPT_PATH}"
    continue
  fi

  for PHASE in qa mt; do
    if [ "$PHASE" = "qa" ]; then
      EVAL_DATA="$QA_EVAL"
    else
      EVAL_DATA="$MT_EVAL"
    fi

    JOB_IDX=$(( JOB_IDX + 1 ))
    GPU=$(( (JOB_IDX - 1) % NUM_GPUS ))
    RUN_NAME="${LABEL}__${PHASE}_eval"
    OUT_DIR="${RESULTS_DIR}/${RUN_NAME}"

    # Wait for the job that was previously assigned to this GPU slot to finish
    # before we reuse the slot (keeps at most NUM_GPUS jobs running at once).
    PREV=$(( JOB_IDX - NUM_GPUS - 1 ))
    if [ "$PREV" -ge 0 ] && [ -n "${PIDS[$PREV]:-}" ]; then
      wait "${PIDS[$PREV]}"
    fi

    run_eval "$GPU" "$RUN_NAME" "$CKPT_PATH" "$EVAL_DATA" "$OUT_DIR" "$JOB_IDX" &
    PIDS+=($!)
  done
done

# Wait for all remaining background jobs
echo ""
echo "Waiting for all evals to finish..."
wait
echo "All evals complete."

# =============================================================================
# Print results summary from logs
# =============================================================================
echo ""
echo "=========================================="
echo "Summary — perplexity extracted from logs"
echo "=========================================="
printf "%-40s %-14s %-14s\n" "Run" "QA ppl" "MT ppl"
printf "%-40s %-14s %-14s\n" "---" "------" "------"

for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r _ LABEL <<< "$ENTRY"
  LABEL=$(echo "$LABEL" | xargs)

  QA_LOG="${RESULTS_DIR}/${LABEL}__qa_eval/eval.log"
  MT_LOG="${RESULTS_DIR}/${LABEL}__mt_eval/eval.log"

  QA_PPL=$(grep -oP "perplexity[=: ]+\K[0-9.]+" "$QA_LOG" 2>/dev/null | tail -1 || echo "N/A")
  MT_PPL=$(grep -oP "perplexity[=: ]+\K[0-9.]+" "$MT_LOG" 2>/dev/null | tail -1 || echo "N/A")

  printf "%-40s %-14s %-14s\n" "$LABEL" "$QA_PPL" "$MT_PPL"
done

echo ""
echo "Full logs: ${RESULTS_DIR}"
echo "Finished at: $(date)"
echo "=========================================="
