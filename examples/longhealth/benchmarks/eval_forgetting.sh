# Forgetting Eval: Phase 1 retention + Phase 2 acquisition for longhealth.
#
# For each Phase 2 checkpoint listed in EXPERIMENTS, this script runs
# train/eval_forgetting.py twice:
#   1. Against the Phase 1 eval parquet (patients 1-10)  → measures forgetting
#   2. Against the Phase 2 eval parquet (patients 11-20) → measures acquisition
#
# Jobs are dispatched round-robin across NUM_GPUS GPUs so all GPUs run in
# parallel (up to NUM_GPUS concurrent evals at a time).
#
# Results (perplexity) are logged to wandb and written to
#   $CARTRIDGES_OUTPUT_DIR/longhealth_forgetting_eval_<pid>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"

# TODO: set these to the actual eval parquets for longhealth when available.
# P1_EVAL should cover patients 1-10 (Phase 1 task — measures forgetting).
# P2_EVAL should cover patients 11-20 (Phase 2 task — measures acquisition).
P1_EVAL="${P1_EVAL:-/localhome/local-triv/gated-continual-cartridges/data/longhealth/eval/patients_01_to_10.parquet}"    # patients 1-10
P2_EVAL="${P2_EVAL:-/localhome/local-triv/gated-continual-cartridges/data/longhealth/eval/patients_11_to_20.parquet}"    # patients 11-20

CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"
RESULTS_DIR="${CARTRIDGES_OUTPUT_DIR}/longhealth_forgetting_eval_${SLURM_JOB_ID:-$$}"
mkdir -p "${RESULTS_DIR}"

# =============================================================================
# Experiment definitions
# Format: "CHECKPOINT_PATH | RUN_LABEL"
#
# Fill in the checkpoint paths from your Phase 2 sparse training runs.
# Example:
#   "${CARTRIDGES_OUTPUT_DIR}/2026-06-XX-XX-XX-XX-continual/<uuid>/cache_last.pt | phase2_freeze"
# =============================================================================
EXPERIMENTS=(
  # TODO: add checkpoint paths once Phase 2 sparse runs are complete
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual/<uuid>/cache_last.pt | phase2_soft"
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual/<uuid>/cache_last.pt | phase2_hard"
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual/<uuid>/cache_last.pt | phase2_freeze"
  # "${CARTRIDGES_OUTPUT_DIR}/YYYY-MM-DD-HH-MM-SS-continual/<uuid>/cache_last.pt | phase2_decouple"
)

echo "=========================================="
echo "LongHealth Forgetting Eval"
echo "Phase 1 retention (p1-10) + Phase 2 acquisition (p11-20)"
echo "=========================================="
echo "Host=$(hostname)"
echo "Started at: $(date)"
echo "Results dir: ${RESULTS_DIR}"
echo "Experiments: ${#EXPERIMENTS[@]} checkpoints × 2 eval sets"
echo "Parallel GPUs: ${NUM_GPUS}"
echo ""

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
  echo "WARNING: No experiments defined — edit the EXPERIMENTS array in this script"
  echo "  and set PHASE1_CACHE_PATH entries to your Phase 2 checkpoint paths."
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
LABELS=()   # matching labels for summary (populated below)

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
  python3 "${CARTRIDGES_DIR}/examples/shared/evaluate/longhealth_forgetting.py" \
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

  for PHASE in p1-10 p11-20; do
    if [ "$PHASE" = "p1-10" ]; then
      EVAL_DATA="$P1_EVAL"
    else
      EVAL_DATA="$P2_EVAL"
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
printf "%-40s %-14s %-14s\n" "Run" "P1-10 ppl" "P11-20 ppl"
printf "%-40s %-14s %-14s\n" "---" "---------" "----------"

for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r _ LABEL <<< "$ENTRY"
  LABEL=$(echo "$LABEL" | xargs)

  P1_LOG="${RESULTS_DIR}/${LABEL}__p1-10_eval/eval.log"
  P2_LOG="${RESULTS_DIR}/${LABEL}__p11-20_eval/eval.log"

  P1_PPL=$(grep -oP "perplexity[=: ]+\K[0-9.]+" "$P1_LOG" 2>/dev/null | tail -1 || echo "N/A")
  P2_PPL=$(grep -oP "perplexity[=: ]+\K[0-9.]+" "$P2_LOG" 2>/dev/null | tail -1 || echo "N/A")

  printf "%-40s %-14s %-14s\n" "$LABEL" "$P1_PPL" "$P2_PPL"
done

echo ""
echo "Full logs: ${RESULTS_DIR}"
echo "Finished at: $(date)"
echo "=========================================="
