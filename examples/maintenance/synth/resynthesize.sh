#!/usr/bin/env bash
# Refill the self-study phases that died mid-run, then regenerate QuALITY at the
# raised completion cap.
#
#   FinQA   p02-p05   died on NaN logprobs -> needs the NaN-safe server
#   TechQA  p02-p05   died on 44.5k-token prompts -> needs 64k context
#   QuALITY p01-p05   complete, but 48.6% of answers hit the old 1024 cap
#
# Each dataset script starts and stops its own vLLM (the three need different
# context lengths), so the stages run strictly in series on one port.
#
# Resilience: a long run can lose its engine to a transient CUDA fault, which
# 500s every later request and fails the phase. One such fault previously idled
# the box for hours, so a stage now retries with a fresh server, only the phases
# still short of 8192 rows are re-requested, and a stage that exhausts its
# retries no longer cancels the stages behind it.
#
#   tmux new -d -s synth-repair 'bash examples/maintenance/synth/resynthesize.sh'
#   tmux attach -t synth-repair

set -uo pipefail

export CARTRIDGES_DIR="${CARTRIDGES_DIR:-/localhome/local-triv/trivo-explore-research-work}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
cd "$CARTRIDGES_DIR"

# GPUs 0-2 are the allocation for this job; DP_SIZE must match the device count.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export DP_SIZE="${DP_SIZE:-3}"
ATTEMPTS="${ATTEMPTS:-4}"
TARGET_ROWS="${TARGET_ROWS:-8192}"

LOGDIR="${LOGDIR:-/localhome/local-triv/outputs_synth_logs}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$LOGDIR/resynthesize_${STAMP}.log"
PY="$CARTRIDGES_DIR/.venv/bin/python"
FAILED=""

log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

rows() {  # rows <dataset> <phase> -> artifact row count, 0 if absent
  "$PY" - "$1" "$2" <<'EOF'
import sys, os
import pyarrow.parquet as pq
p = f"data/{sys.argv[1]}/synth/p0{sys.argv[2]}/self_study-n8192/artifact/dataset.parquet"
try:
    print(pq.ParquetFile(p).metadata.num_rows if os.path.exists(p) else 0)
except Exception:
    print(0)
EOF
}

pending() {  # pending <dataset> <phases...> -> those still short of TARGET_ROWS
  local ds="$1"; shift
  local out=""
  for ph in "$@"; do
    [ "$(rows "$ds" "$ph")" -lt "$TARGET_ROWS" ] && out="$out $ph"
  done
  echo "${out# }"
}

report() {  # report <dataset> <phases...>
  local ds="$1"; shift
  for ph in "$@"; do
    local n; n="$(rows "$ds" "$ph")"
    log "    $ds p0$ph -> $n / $TARGET_ROWS rows $([ "$n" -ge "$TARGET_ROWS" ] && echo OK || echo SHORT)"
  done
}

stage() {  # stage <dataset> <script> <phases>
  local ds="$1" script="$2" phases="$3"
  log "=============================================================="
  log "STAGE $ds  requested=[$phases]  gpus=$CUDA_VISIBLE_DEVICES dp=$DP_SIZE"
  log "=============================================================="
  local attempt todo
  for attempt in $(seq 1 "$ATTEMPTS"); do
    todo="$(pending "$ds" $phases)"
    if [ -z "$todo" ]; then
      log "STAGE $ds complete"
      report "$ds" $phases
      return 0
    fi
    log "attempt $attempt/$ATTEMPTS — phases still short: [$todo]"
    PHASES="$todo" bash "$script" 2>&1 | tee -a "$DRIVER_LOG"
    log "attempt $attempt finished (script rc=${PIPESTATUS[0]})"
    # Trust row counts over exit codes: a phase can fail loudly at the very end
    # and still have written every checkpoint it needed.
    sleep 20
  done
  todo="$(pending "$ds" $phases)"
  if [ -z "$todo" ]; then
    log "STAGE $ds complete"
  else
    log "STAGE $ds EXHAUSTED $ATTEMPTS attempts — still short: [$todo] (continuing to next stage)"
    FAILED="$FAILED $ds"
  fi
  report "$ds" $phases
}

log "driver log: $DRIVER_LOG"
log "start: $(date)"

stage finqa  examples/finqa/synthesize/self_study.sh  "2 3 4 5"
stage techqa examples/techqa/synthesize/self_study.sh "2 3 4 5"

# QuALITY is complete at the old cap, so pending() would report nothing to do.
# Retire the old artifacts first. The backup is the ONLY copy of the cap-1024
# data, so each artifact is verified in the backup before the original is
# removed; anything that fails to copy is left untouched.
QBK="/localhome/local-triv/quality_cap1024_backup_${STAMP}"
log "=============================================================="
log "retiring cap-1024 QuALITY artifacts -> $QBK"
for ph in 1 2 3 4 5; do
  d="data/quality/synth/p0${ph}/self_study-n8192"
  src="$d/artifact/dataset.parquet"
  [ -f "$src" ] || { log "    p0${ph} has no artifact, nothing to retire"; continue; }
  mkdir -p "$QBK/p0${ph}"
  cp -a "$src" "$QBK/p0${ph}/dataset.parquet"
  n="$("$PY" -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$QBK/p0${ph}/dataset.parquet').metadata.num_rows)" 2>/dev/null || echo 0)"
  if [ "$n" -ge "$TARGET_ROWS" ]; then
    log "    p0${ph} backed up ($n rows) — clearing original"
    rm -f "$d"/checkpoints/batch_*.parquet "$src"
  else
    log "    p0${ph} BACKUP VERIFY FAILED ($n rows) — leaving original in place"
  fi
done

stage quality examples/quality/synthesize/self_study.sh "1 2 3 4 5"

log "=============================================================="
log "ALL STAGES DONE${FAILED:+ (incomplete:$FAILED)}"
report finqa 2 3 4 5
report techqa 2 3 4 5
report quality 1 2 3 4 5
log "finish: $(date)"
[ -z "$FAILED" ]
