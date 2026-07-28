# Forgetting Eval: Phase 1 retention + Phase 2 acquisition for qasper.
#
# For each Phase 2 checkpoint listed in EXPERIMENTS, this script runs
# train/eval_forgetting.py twice:
#   1. Against the Phase 1 eval parquet (QA task)  → measures forgetting
#   2. Against the Phase 2 eval parquet (MT task)  → measures acquisition
#
# Jobs are dispatched round-robin across NUM_GPUS GPUs so all GPUs run in
# parallel (up to NUM_GPUS concurrent evals at a time).
#
# Results (perplexity) are logged to wandb and written to
#   $CARTRIDGES_OUTPUT_DIR/qasper_forgetting_eval_<pid>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-4}"

# All wandb runs from this batch land under one group so the analysis
# pipeline (sparse_continual_dynamics_investigation/scripts/01b_pull_forgetting_eval.py)
# can pull them with a single (entity, project, group) filter.
WANDB_GROUP="${WANDB_GROUP:-qasper - [granularity x forgetting/learning]}"

# Phase 1 task: QA — used to measure forgetting.
# Phase 2 task: MT — used to measure acquisition.
QA_EVAL="${QA_EVAL:-${CARTRIDGES_DIR}/examples/qasper2/qasper_eval_QA.parquet}"
MT_EVAL="${MT_EVAL:-${CARTRIDGES_DIR}/examples/qasper2/qasper_eval_MT.parquet}"

CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"
RESULTS_DIR="${CARTRIDGES_OUTPUT_DIR}/qasper_forgetting_eval_key-value_bsize32_${SLURM_JOB_ID:-$$}"
mkdir -p "${RESULTS_DIR}"

# =============================================================================
# Experiment definitions
# Format: "CHECKPOINT_PATH | RUN_LABEL"
#
# Fill in the checkpoint paths from your Phase 2 sparse training runs.
# Example:
#   "${CARTRIDGES_OUTPUT_DIR}/2026-06-XX-XX-XX-XX-continual_sparse/<uuid>/cache_last.pt | phase2_freeze"
# =============================================================================
EXPERIMENTS=(
  # TODO: add checkpoint paths once Phase 2 sparse runs are complete
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-initial_sparse_per-head/<uuid>/cache_last.pt | baseline"
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual_sparse-per-head-top-64/<uuid>/cache_last.pt | per-head-top-64"
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual_sparse-per-head-top-128/<uuid>/cache_last.pt | per-head-top-128"
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual_sparse-per-head-top-256/<uuid>/cache_last.pt | per-head-top-256"
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual_sparse-per-head-top-512/<uuid>/cache_last.pt | per-head-top-512"
  # "${CARTRIDGES_OUTPUT_DIR}/qasper-initial-per-head-all-reduce/51f1e2fb-321a-4cee-8796-9a826fb6681d/cache-step534.pt | baseline_per-head"
  # "${CARTRIDGES_OUTPUT_DIR}/qasper-initial-per-layer-all-reduce/d8103e75-4886-47a0-8af0-286ca4bec665/cache-step534.pt | baseline_per-layer"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-head-top-64-key-value_all-reduce/55828101-8aa7-4c3b-b892-cc40e551cea8/cache-step564.pt | qasper-key-value-per-head-top-64"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-head-top-128-key-value_all-reduce/08d4e035-794e-4759-880a-e2c102f73e06/cache-step564.pt | qasper-key-value-per-head-top-128"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-head-top-256-key-value_all-reduce/7136f9dd-666a-403b-9f53-488ad336401d/cache-step564.pt | qasper-key-value-per-head-top-256"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-head-top-512-key-value_all-reduce/45e9e79f-3d0d-4b7e-90f0-a0780cf4f2c3/cache-step564.pt | qasper-key-value-per-head-top-512"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-layer-top-64-key-value_all-reduce/693375b6-2236-4b02-9d7e-b64875a288b9/cache-step564.pt | qasper-key-value-per-layer-top-64"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-layer-top-128-key-value_all-reduce/8b6e44ec-1a7a-4ea7-8b6d-ce4fef400474/cache-step564.pt | qasper-key-value-per-layer-top-128"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-layer-top-256-key-value_all-reduce/24c04e5c-a3fe-4ea3-8f64-6730ebbb5940/cache-step564.pt | qasper-key-value-per-layer-top-256"
  "${CARTRIDGES_OUTPUT_DIR}/qasper-per-layer-top-512-key-value_all-reduce/e0730349-5065-4fea-83e7-8c1e93a69a1f/cache-step564.pt | qasper-key-value-per-layer-top-512"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-08-49-29-continual_sparse-qasper-per-layer-top-64-value-only_all-reduce/e7d58f58-c364-4067-87cb-49f97ccc607e/cache-step282.pt | qasper-value-only-per-layer-top-64"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-08-49-33-continual_sparse-qasper-per-layer-top-128-value-only_all-reduce/8395d484-67c8-4afe-b0f9-4c70919c76ac/cache-step282.pt | qasper-value-only-per-layer-top-128"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-10-01-25-continual_sparse-qasper-per-layer-top-256-value-only_all-reduce/14ba2a6c-47f3-42bc-aec7-522cdff4f219/cache-step282.pt | qasper-value-only-per-layer-top-256"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-10-01-45-continual_sparse-qasper-per-layer-top-512-value-only_all-reduce/63ead1fb-cac6-4d0b-929e-5901978b318d/cache-step282.pt | qasper-value-only-per-layer-top-512"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-04-58-20-continual_sparse-qasper-per-head-top-64-value-only_all-reduce/0a962537-6e01-4d21-9a4d-5ffcdb45e7ea/cache-step282.pt | qasper-value-only-per-head-top-64"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-04-59-17-continual_sparse-qasper-per-head-top-128-value-only_all-reduce/f8e38da6-e293-4246-a873-f80f84b55d65/cache-step282.pt | qasper-value-only-per-head-top-128"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-06-11-28-continual_sparse-qasper-per-head-top-256-value-only_all-reduce/9d046bcf-e057-442d-8878-7aea78bd51c2/cache-step282.pt | qasper-value-only-per-head-top-256"
  # "${CARTRIDGES_OUTPUT_DIR}/2026-06-18-06-11-48-continual_sparse-qasper-per-head-top-512-value-only_all-reduce/4a298966-7c47-439e-b90b-e581825a034a/cache-step282.pt | qasper-value-only-per-head-top-512"
)

echo "=========================================="
echo "Qasper Forgetting Eval"
echo "Phase 1 retention (QA) + Phase 2 acquisition (MT)"
echo "=========================================="
echo "Host=$(hostname)"
echo "Started at: $(date)"
echo "Results dir: ${RESULTS_DIR}"
echo "Experiments: ${#EXPERIMENTS[@]} checkpoints × 2 eval sets"
echo "Parallel GPUs: ${NUM_GPUS}"
echo "WandB group:   ${WANDB_GROUP:-<unset>}"
echo ""

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
  echo "WARNING: No experiments defined — edit the EXPERIMENTS array in this script"
  echo "  and set entries to your Phase 2 checkpoint paths."
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

# Activate the repo virtual environment (pydrantic + cartridges live in .venv)
echo "Activating .venv..."
source "$CARTRIDGES_DIR/.venv/bin/activate"
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
