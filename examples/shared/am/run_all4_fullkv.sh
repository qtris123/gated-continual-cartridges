#!/usr/bin/env bash
# Full-KV continual-AM 5x5 grid for all four benchmarks, one benchmark per GPU.
#
# "Full KV" = the write updates EVERY cartridge slot instead of the top-t the
# attention/TF-IDF ranker selects. This is a single-knob change: the recipe's
# slots.top_t is 512 (>= the 511 trainable slots), and the ranker caps at
# k = min(top_t, n_tokens), so every slot lands in `selected_full`. That one
# index set drives BOTH the key rewrite (key_mode=highest_attention + reposition)
# and the value solve, so K and V are both refit on all slots. Everything else
# (rope rebake @ 5e6, beta off, delta_weight 1e-2) is the ropefix recipe verbatim.
#
# Each dataset owns one GPU for both its writes and its five per-stage evals, so
# the four chains run fully in parallel without contending for a device. Each
# chain is stage-resumable under its own tag.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${CARTRIDGES_PYTHON:-$ROOT/.venv/bin/python}"
export CARTRIDGES_PYTHON="$PY"
RECIPE="${RECIPE:-$ROOT/outputs/recipes/fullkv_topt512.yaml}"
TAG="${TAG:-delta_ha_b0_idf0_ropefix_p01D_fullkv512}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# dataset : gpu : p01 cache
QASPER_P01="$ROOT/outputs/hf_p01/qasper/cache_last.pt"
QUALITY_P01="$ROOT/outputs/hf_p01/quality/cache_last.pt"
FINQA_P01="$(ls "$ROOT"/outputs/finqa_p1_ropefix/*/*/cache-step0.pt | head -1)"
TECHQA_P01="$(ls "$ROOT"/outputs/techqa_p1_ropefix/*/*/cache-step0.pt | head -1)"

declare -a JOBS=(
  "qasper 0 $QASPER_P01"
  "quality 1 $QUALITY_P01"
  "finqa 2 $FINQA_P01"
  "techqa 3 $TECHQA_P01"
)

launch() {
  local ds="$1" gpu="$2" p01="$3"
  local log="$LOGDIR/${ds}_fullkv_${TS}.log"
  echo "[$ds] gpu=$gpu p01=$p01"
  echo "[$ds] log=$log"
  "$PY" "$ROOT/examples/shared/am/run_chain.py" \
    --dataset "$ds" \
    --p01-cache "$p01" \
    --tag "$TAG" \
    --recipe-config "$RECIPE" \
    --gpu "$gpu" \
    --eval-gpus "$gpu" \
    "${EXTRA_ARGS[@]}" \
    >"$log" 2>&1 &
  echo "[$ds] pid=$! (gpu $gpu)"
}

EXTRA_ARGS=("$@")

echo "=== full-KV 5x5 for all 4 benchmarks | recipe=$RECIPE | tag=$TAG ==="
pids=()
for spec in "${JOBS[@]}"; do
  read -r ds gpu p01 <<<"$spec"
  launch "$ds" "$gpu" "$p01"
  pids+=($!)
done

echo "=== launched ${#pids[@]} chains; waiting ==="
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
echo "=== all chains finished (aggregate rc=$rc) at $(date) ==="
exit "$rc"
