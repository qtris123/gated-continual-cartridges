#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=forgetting_eval
#SBATCH --output=forgetting_eval_%j.out
#SBATCH --error=forgetting_eval_%j.err

set -euo pipefail

# =============================================================================
# Forgetting Eval: Phase 1 retention + Phase 2 acquisition
#
# Runs eval_forgetting.py on each momentum-ablation checkpoint against both
# the QA eval parquet (measures Phase 1 retention / forgetting) and the MT
# eval parquet (measures Phase 2 acquisition).
#
# Results are logged to wandb and written to CARTRIDGES_OUTPUT_DIR/forgetting_eval_*/
# =============================================================================

export CARTRIDGES_DIR=/home/vo43/cartridges

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"

QA_EVAL="${CARTRIDGES_DIR}/examples/qasper2/qasper_eval_QA.parquet"   # Phase 1 task
MT_EVAL="${CARTRIDGES_DIR}/examples/qasper2/qasper_eval_MT.parquet"   # Phase 2 task

CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"
RESULTS_DIR="${CARTRIDGES_OUTPUT_DIR}/forgetting_eval_key-value_sgd_${SLURM_JOB_ID:-$$}"
mkdir -p "${RESULTS_DIR}"

# =============================================================================
# Experiment definitions
# Format: "CHECKPOINT_PATH | RUN_LABEL"
#
# Confirmed from config.yaml (momentum_masking field is the ground truth):
#   2026-06-03-21-19-48  10107dc6  momentum=hard  name=qasper_phase2_soft  ← mislabeled, skip
#   2026-06-03-23-13-10  585fc341  momentum=soft  name=qasper_phase2_soft
#   2026-06-04-00-39-47  f553ccb5  momentum=hard  name=qasper_phase2_hard
#   2026-06-04-02-08-09  01235943  momentum=freeze name=qasper_phase2_freeze
#   2026-06-04-03-32-16  cd74de7d  momentum=decouple name=qasper_phase2_decouple
# =============================================================================
EXPERIMENTS=(
  #"${CARTRIDGES_OUTPUT_DIR}/2026-05-28-03-57-51-initial/42a82b24-75f3-4bd8-b77f-447f3bb5d07f/cache_last.pt | phase1_baseline"
  "${CARTRIDGES_OUTPUT_DIR}/2026-06-07-13-46-15-continual/57e8ed37-9bc6-48b4-9909-ba83c1a4927e/cache_last.pt | phase2_soft"
  "${CARTRIDGES_OUTPUT_DIR}/2026-06-10-20-28-33-continual/18aa9bf6-fd76-4a90-a48b-554a1b20efa5/cache_last.pt | phase2_hard"
  "${CARTRIDGES_OUTPUT_DIR}/2026-06-07-15-16-40-continual/d5d52a4c-6946-46b3-b5c8-6f3718479725/cache_last.pt | phase2_freeze"
  "${CARTRIDGES_OUTPUT_DIR}/2026-06-07-04-34-58-continual/91f5e1dc-756d-4925-98ab-70bffa20f1c1/cache_last.pt | phase2_decouple"
)

echo "=========================================="
echo "Forgetting Eval — Phase 1 QA + Phase 2 MT"
echo "=========================================="
echo "JobID=${SLURM_JOB_ID:-local}"
echo "Node=$(hostname)"
echo "Started at: $(date)"
echo "Results dir: ${RESULTS_DIR}"
echo "Experiments: ${#EXPERIMENTS[@]} checkpoints × 2 eval sets"
echo ""

# --- Load modules ---
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

# --- Activate conda ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cartridges
echo "Python: $(which python3)"
echo ""

# =============================================================================
# Main eval loop
# =============================================================================
TOTAL=$(( ${#EXPERIMENTS[@]} * 2 ))
IDX=0

for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r CKPT_PATH LABEL <<< "$ENTRY"
  CKPT_PATH=$(echo "$CKPT_PATH" | xargs)
  LABEL=$(echo "$LABEL" | xargs)

  if [ ! -f "$CKPT_PATH" ]; then
    echo "WARNING: checkpoint not found, skipping: ${CKPT_PATH}"
    continue
  fi

  # --- Phase 1 eval (QA task — measures forgetting) ---
  IDX=$((IDX + 1))
  RUN_NAME="${LABEL}__qa_eval_key-value_sgd"
  OUT_DIR="${RESULTS_DIR}/${RUN_NAME}"
  mkdir -p "$OUT_DIR"

  echo "══════════════════════════════════════════"
  echo "  [${IDX}/${TOTAL}] ${RUN_NAME}"
  echo "  Checkpoint: ${CKPT_PATH}"
  echo "  Eval: QA (Phase 1)"
  echo "══════════════════════════════════════════"

  CHECKPOINT_PATH="$CKPT_PATH" \
  EVAL_DATA_PATH="$QA_EVAL" \
  MODEL_NAME="$MODEL_NAME" \
  RUN_NAME="$RUN_NAME" \
  BATCH_SIZE="$BATCH_SIZE" \
  python3 "${CARTRIDGES_DIR}/examples/qasper2/train/eval_forgetting.py" \
    2>&1 | tee "${OUT_DIR}/eval_qa.log"

  echo "  Done: ${RUN_NAME}"

  # --- Phase 2 eval (MT task — measures acquisition) ---
  IDX=$((IDX + 1))
  RUN_NAME="${LABEL}__mt_eval_key-value_sgd"
  OUT_DIR="${RESULTS_DIR}/${RUN_NAME}"
  mkdir -p "$OUT_DIR"

  echo "══════════════════════════════════════════"
  echo "  [${IDX}/${TOTAL}] ${RUN_NAME}"
  echo "  Checkpoint: ${CKPT_PATH}"
  echo "  Eval: MT (Phase 2)"
  echo "══════════════════════════════════════════"

  CHECKPOINT_PATH="$CKPT_PATH" \
  EVAL_DATA_PATH="$MT_EVAL" \
  MODEL_NAME="$MODEL_NAME" \
  RUN_NAME="$RUN_NAME" \
  BATCH_SIZE="$BATCH_SIZE" \
  python3 "${CARTRIDGES_DIR}/examples/qasper2/train/eval_forgetting.py" \
    2>&1 | tee "${OUT_DIR}/eval_mt.log"

  echo "  Done: ${RUN_NAME}"
  echo ""
done

# =============================================================================
# Print results summary from logs
# =============================================================================
echo ""
echo "=========================================="
echo "Summary — perplexity extracted from logs"
echo "=========================================="
printf "%-35s %-12s %-12s\n" "Run" "QA ppl" "MT ppl"
printf "%-35s %-12s %-12s\n" "---" "------" "------"

for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r _ LABEL <<< "$ENTRY"
  LABEL=$(echo "$LABEL" | xargs)

  QA_LOG="${RESULTS_DIR}/${LABEL}__qa_eval/eval_qa.log"
  MT_LOG="${RESULTS_DIR}/${LABEL}__mt_eval/eval_mt.log"

  QA_PPL=$(grep -oP "perplexity[=: ]+\K[0-9.]+" "$QA_LOG" 2>/dev/null | tail -1 || echo "N/A")
  MT_PPL=$(grep -oP "perplexity[=: ]+\K[0-9.]+" "$MT_LOG" 2>/dev/null | tail -1 || echo "N/A")

  printf "%-35s %-12s %-12s\n" "$LABEL" "$QA_PPL" "$MT_PPL"
done

echo ""
echo "Full logs: ${RESULTS_DIR}"
echo "Finished at: $(date)"
echo "=========================================="
