#!/usr/bin/env bash
# Sweep over TOP_T sparsity values by running train_continual_sparse.sh once per value.
# Each run is fully independent — a new output directory and W&B run are created automatically.
#
# Usage:
#   bash examples/longhealth/sweep_top_t.sh
#
# Override any variable from train_continual_sparse.sh via the environment, e.g.:
#   NUM_GPUS=2 bash examples/longhealth/sweep_top_t.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/../pipelines/train_continual_sparse.sh"

TOP_T_VALUES=(64 128 256 512)

echo "=========================================="
echo "TOP_T sweep: ${TOP_T_VALUES[*]}"
echo "Started at: $(date)"
echo "=========================================="
echo ""

for TOP_T in "${TOP_T_VALUES[@]}"; do
    echo "------------------------------------------"
    echo "Starting experiment: TOP_T=$TOP_T  ($(date))"
    echo "------------------------------------------"

    TOP_T="$TOP_T" \
    RUN_NAME="" \
        bash "$TRAIN_SCRIPT"

    echo ""
    echo "Finished TOP_T=$TOP_T at $(date)"
    echo ""
done

echo "=========================================="
echo "All TOP_T experiments complete: $(date)"
echo "=========================================="
