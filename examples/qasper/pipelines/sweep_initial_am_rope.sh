#!/usr/bin/env bash
# Single-variable sweep over how Phase-1 AM compaction handles absolute position.
#
# Phase-1 prefills every QASPER document from position 0 and concatenates, then
# counter-rotates the selected keys onto sequential cartridge slots. That rotation
# lands on the slot only when it uses the base the keys were baked with, and the
# driver's default is the historical 1e4 while Qwen3-4B-Instruct-2507 is 5e6
# (MECH-003 / MECH-005 fixed this on the continual side only).
#
# Arms, holding everything else fixed:
#   S  shipped default            rebake@1e4,  per-doc positions, beta on
#   A  beta removed               rebake@1e4,  per-doc positions, beta off
#   B  no positional correction   no rebake,   per-doc positions, beta off
#   C  the fix                    rebake@5e6,  per-doc positions, beta off
#   D  fix + unique positions     rebake@5e6,  global positions,  beta off
#   E  does the fix rescue beta?  rebake@5e6,  per-doc positions, beta on
#
# S vs A isolates beta; A vs C isolates the rotary base; B is the floor that says
# whether correcting at all beats leaving the keys where the teacher put them.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWEEP_ROOT="${SWEEP_ROOT:-$CARTRIDGES_DIR/outputs/p1_rope_sweep}"
mkdir -p "$SWEEP_ROOT/logs"

# The launcher calls bare `python`; make sure it is the project interpreter.
PY_BIN_DIR="${PY_BIN_DIR:-/localhome/local-triv/miniforge3/envs/cartridges/bin}"
[ -d "$PY_BIN_DIR" ] && export PATH="$PY_BIN_DIR:$PATH"

# --- held constant across every arm -------------------------------------------
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
export NUM_TOKENS=512
export KEY_SELECT=highest_attention
export RIDGE_SCALE=spectral
export RIDGE_LAMBDA=1e-4
export GRANULARITY=per_head
export MAX_REF_BATCHES=50
export MAX_QUERIES_PER_HEAD=64
export MAX_REF_EXAMPLES=256
export QUERIES_PER_BATCH=all_tokens
export STRIP_REF_SYSTEM_PROMPT=1
export MAX_TEACHER_TOKENS=""
export SEED=42
# bg_stats feed a later continual phase; this sweep reads perplexity, so keep the
# collection cheap but identical across arms.
export NUM_BG_BATCHES="${NUM_BG_BATCHES:-64}"
export EVAL_P1_PATH="$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_QA.parquet"
export EVAL_P2_PATH="$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_MT.parquet"
export EVAL_P1_NAME=qa
export EVAL_P2_NAME=mt
export WANDB_DISABLED="${WANDB_DISABLED:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# arm_name  rebake  rope_theta  global_positions  beta
ARMS=(
  "S_shipped_b1_rebake1e4     1  10000.0    0  1"
  "A_b0_rebake1e4             1  10000.0    0  0"
  "B_b0_norebake              0  10000.0    0  0"
  "C_b0_rebake5e6             1  5000000.0  0  0"
  "D_b0_rebake5e6_globalpos    1  5000000.0  1  0"
  "E_b1_rebake5e6             1  5000000.0  0  1"
)

ONLY="${ONLY:-}"
echo "=== Phase-1 RoPE sweep: ${#ARMS[@]} arms -> $SWEEP_ROOT ==="
date

for spec in "${ARMS[@]}"; do
  read -r arm rebake theta gpos beta <<<"$spec"
  if [ -n "$ONLY" ] && [[ "$arm" != *"$ONLY"* ]]; then
    echo "--- skip $arm (ONLY=$ONLY)"
    continue
  fi

  log="$SWEEP_ROOT/logs/$arm.log"
  echo
  echo "=== ARM $arm  rebake=$rebake theta=$theta global_pos=$gpos beta=$beta ==="
  echo "    log: $log"

  REBAKE_KEY_POSITIONS="$rebake" \
  AM_ROPE_THETA="$theta" \
  GLOBAL_TEACHER_POSITIONS="$gpos" \
  ENABLE_BETA="$beta" \
  RUN_NAME="p1rope_$arm" \
  CARTRIDGES_OUTPUT_DIR="$SWEEP_ROOT/$arm" \
    bash "$SCRIPT_DIR/train_initial_am_compaction.sh" >"$log" 2>&1
  rc=$?

  if [ $rc -eq 0 ]; then
    echo "    OK  $(date +%H:%M:%S)"
    grep -E '^- (qa|mt|QA) eval|recon MSE|Per-doc prefill|Each doc prefilled|Selected keys|Keys installed' \
      "$SWEEP_ROOT/$arm"/*/*/SUMMARY.md 2>/dev/null | sed 's/^/    /'
  else
    echo "    FAILED rc=$rc — tail of log:"
    tail -15 "$log" | sed 's/^/    /'
  fi
done

echo
echo "=== sweep done ==="
date
