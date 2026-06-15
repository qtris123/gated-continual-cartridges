#!/usr/bin/env bash
# Submit all 8 momentum × freeze_keys experiments as separate SLURM jobs.
#
# Usage (from the scripts/ directory):
#   bash launch_all_experiments.sh
#
# Each job inherits the full train_continual.sh logic.
# Override any shared parameter with an env var before calling this script, e.g.:
#   EPOCHS=5 bash launch_all_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_continual.sh"

# Shared overrides — set these before calling the script if you want different defaults
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-/home/vo43/cartridges/outputs/2026-05-28-03-57-51-initial/42a82b24-75f3-4bd8-b77f-447f3bb5d07f/cache-step534.pt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/scratch/scholar/vo43/qasper-MT_8192_off-policy.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/vo43/cartridges/examples/qasper2/qasper_eval_MT.parquet}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2.0}"
TOP_T="${TOP_T:-512}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-15}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
NUM_TOKENS="${NUM_TOKENS:-1024}"

MOMENTUM_MODES=(soft hard freeze decouple)
FREEZE_KEYS_VALUES=(1 0)

echo "Submitting 8 experiments..."
echo ""

for MODE in "${MOMENTUM_MODES[@]}"; do
    for FK in "${FREEZE_KEYS_VALUES[@]}"; do
        FK_LABEL=$([ "$FK" = "1" ] && echo "frozen-keys" || echo "free-keys")
        RUN_NAME="qasper_phase2_${MODE}_${FK_LABEL}"
        OUT_FILE="${SCRIPT_DIR}/${RUN_NAME}.out"
        ERR_FILE="${SCRIPT_DIR}/${RUN_NAME}.err"

        JOB_ID=$(sbatch \
            --job-name="$RUN_NAME" \
            --output="$OUT_FILE" \
            --error="$ERR_FILE" \
            --export=ALL,\
MOMENTUM_MASKING="$MODE",\
FREEZE_KEYS="$FK",\
RUN_NAME="$RUN_NAME",\
PHASE1_CACHE_PATH="$PHASE1_CACHE_PATH",\
SYNTH_DATA_PATH="$SYNTH_DATA_PATH",\
EVAL_DATA_PATH="$EVAL_DATA_PATH",\
EPOCHS="$EPOCHS",\
LR="$LR",\
TOP_T="$TOP_T",\
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE",\
EVAL_EVERY_N_STEPS="$EVAL_EVERY_N_STEPS",\
SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS",\
NUM_TOKENS="$NUM_TOKENS" \
            "$TRAIN_SCRIPT" \
            | awk '{print $NF}')

        echo "  [$JOB_ID]  momentum=$MODE  freeze_keys=$FK  →  $RUN_NAME"
    done
done

echo ""
echo "All 8 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f <run_name>.err"
