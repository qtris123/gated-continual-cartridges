#!/usr/bin/env bash
# DIAG-CONTENT (W5 VERIFY, board B-ROUTE / B-OVERWRITE)
#
# The claim under attack: that AM's Phase-2 write stores document CONTENT at all.
#
#   arm A  canonical  = MT corpus written, MT evaluated   (REUSED, not re-run:
#                       DIAG-OVERWRITE run dir + DIAG-SEQUENCE per-k curve)
#   arm B  content-free control = QA corpus (Phase-1 topics) written, MT evaluated.
#                       The 16 QA papers contain NO MT content by construction, so any
#                       MT improvement they produce is content-free.
#   arm C  order control        = the SAME 16 MT documents written in REVERSED order.
#                       Content held constant, only the sequence changes. Guards the
#                       "arm B is closer to a no-op because the Phase-1 cartridge was
#                       BUILT from the QA corpus" confound from a second direction:
#                       if the front-loaded gain reappears with a different first-3
#                       documents, the early gain is positional, not content-driven.
#
# NO SOURCE FILE IS EDITED. Imports are pinned to a frozen snapshot of committed code
# (RUNBOOK 9c-bis) because MECH-BETA is editing cartridges/am/{key_select,finetune}.py.
# The reversed-order corpus is a DATA artefact produced by make_reversed_parquet.py.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_content
PY="$REPO/.venv/bin/python"
export PATH="$REPO/.venv/bin:$PATH"          # so the wrapper script's bare `python` is the venv

export CARTRIDGES_DIR="$SNAP"                # driver + package both from committed HEAD
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"

RESDIR="$REPO/research_loop/results/DIAG-CONTENT"
LOGDIR="$RESDIR/evals"
mkdir -p "$LOGDIR"
exec > "$RESDIR/wrapper.log" 2>&1

echo "DIAG_CONTENT_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
echo "SNAPSHOT_PATH=$SNAP"
manifest() { (cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1); }
echo "SNAPSHOT_MANIFEST=$(manifest)"

# ---- claim exactly ONE GPU via flock, hold it for the whole script -----------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 120); do
  for idx in 0 1; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "DIAG_CONTENT_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 30
done
if [ -z "$CLAIMED" ]; then echo "DIAG_CONTENT_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "DIAG_CONTENT_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT"

# ---- MANDATORY pin proof, from /tmp (NOT the repo root) ---------------------
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

PHASE1="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
QA_EVAL="$REPO/data/qasper/eval/qasper_eval_QA.parquet"
MT_EVAL="$REPO/data/qasper/eval/qasper_eval_MT.parquet"

# =============================================================================
# training arms -- CANONICAL config everywhere except SYNTH_DATA_PATH
# =============================================================================
train_arm() {   # label  synth_path  notes
  local label="$1" synth="$2" notes="$3"
  echo "DIAG_CONTENT_TRAIN_${label}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T=32 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE=freeze \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  DELTA_WEIGHT=1e-2 \
  MAX_QUERIES_PER_HEAD=64 \
  MAX_REF_EXAMPLES_PER_DOC=32 \
  AM_EXECUTION_MODE=per_document \
  SAVE_AFTER_EACH_DOCUMENT=1 \
  PHASE1_CACHE_PATH="$PHASE1" \
  SYNTH_DATA_PATH="$synth" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH="$QA_EVAL" \
  EVAL_MT_PATH="$MT_EVAL" \
  RUN_NAME="DIAG-CONTENT_${label}" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-OVERWRITE \
  WANDB_NOTES="$notes" \
  bash "$SNAP/examples/qasper2/scripts/core/train_continual_am_sparse.sh" >"$RESDIR/run_${label}.log" 2>&1
  local rc=$?
  echo "DIAG_CONTENT_TRAIN_${label}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  local rd
  rd=$(grep -oE "Saved to [^ ]+" "$RESDIR/run_${label}.log" | tail -1 | awk '{print $3}')
  if [ -z "$rd" ]; then
    rd=$(grep -oE '"run_dir": "[^"]+"' "$RESDIR/run_${label}.log" | tail -1 | cut -d'"' -f4)
  fi
  echo "DIAG_CONTENT_RUNDIR_${label}=$rd"
  echo "$rd" > "$RESDIR/rundir_${label}.txt"
  echo "DIAG_CONTENT_WANDB_${label}=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$RESDIR/run_${label}.log" | head -1)"
  grep -oE 'Eval loss - [0-9.]+' "$RESDIR/run_${label}.log" | tail -2
  return $rc
}

train_arm armB "$REPO/data/qasper/train/qwen_qasper_QA_task_8192.parquet" \
  "content-free write control: QA-corpus documents, MT eval"
RC_B=$?

train_arm armC "$REPO/data/qasper/train/DIAG-CONTENT_qwen_qasper_MT_task_8192_revorder.parquet" \
  "order control: the same 16 MT documents written in reversed order, MT eval"
RC_C=$?

# =============================================================================
# per-k standalone evals (same convention as DIAG-SEQUENCE, so the curves compare)
# =============================================================================
SUMMARY="$RESDIR/curve.tsv"
: > "$SUMMARY"
printf "arm\tk\tsplit\tloss\tckpt\twandb_url\n" >> "$SUMMARY"

eval_one() {   # arm k split ckpt
  local arm="$1" k="$2" split="$3" ckpt="$4"
  local log="$LOGDIR/eval_${arm}_k${k}_${split}.log"
  : > "$log"
  echo "--- [$(date -Is)] arm=${arm} k=${k} split=${split} ckpt=${ckpt}"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="DIAG-CONTENT_${arm}_k${k}_${split}" \
  WANDB_GROUP=B-OVERWRITE \
  WANDB_DISABLED=0 \
  "$PY" "$SNAP/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 240); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  local loss url
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  echo "RESULT arm=${arm} k=${k} ${split}: loss=${loss:-MISSING} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK 1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

# k=0 anchor (untouched Phase-1) once, as a same-session control
eval_one phase1 0 QA "$PHASE1"
eval_one phase1 0 MT "$PHASE1"

for ARM in armB armC; do
  RD=$(cat "$RESDIR/rundir_${ARM}.txt" 2>/dev/null)
  if [ -z "$RD" ] || [ ! -d "$RD" ]; then echo "NO_RUNDIR_${ARM}"; continue; fi
  for K in 16 1 4 8 12; do
    IDX=$(printf "%03d" $((K-1)))
    CKPT=$(ls "$RD"/cache-after-doc-${IDX}-*.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then echo "MISSING_SNAPSHOT arm=$ARM k=$K idx=$IDX"; continue; fi
    eval_one "$ARM" "$K" MT "$CKPT"
    eval_one "$ARM" "$K" QA "$CKPT"
  done
done

# =============================================================================
# eval-time routing diagnostic: mass_on_S + total cartridge mass, both splits.
# arm A's final cache is included so the comparison is same-session.
# =============================================================================
RD_B=$(cat "$RESDIR/rundir_armB.txt" 2>/dev/null)
RD_C=$(cat "$RESDIR/rundir_armC.txt" 2>/dev/null)
RD_A="$REPO/outputs/2026-07-28-21-43-17-continual_am_sparse/4b0edcab-d1c5-4313-bc43-309d7ff77097"
CK_A=$(ls "$RD_A"/cache-after-doc-015-*.pt 2>/dev/null | head -1)
CK_B=$(ls "$RD_B"/cache-after-doc-015-*.pt 2>/dev/null | head -1)
CK_C=$(ls "$RD_C"/cache-after-doc-015-*.pt 2>/dev/null | head -1)
SPECS=""
[ -n "$CK_A" ] && SPECS="armA:${CK_A}:${RD_A}"
[ -n "$CK_B" ] && SPECS="${SPECS}${SPECS:+,}armB:${CK_B}:${RD_B}"
[ -n "$CK_C" ] && SPECS="${SPECS}${SPECS:+,}armC:${CK_C}:${RD_C}"
echo "DIAG_CONTENT_ROUTE_SPECS=$SPECS"
if [ -n "$SPECS" ]; then
  echo "DIAG_CONTENT_ROUTE_START_EPOCH=$(date +%s) TS=$(date -Is)"
  CACHE_SPECS="$SPECS" \
  EVAL_SPECS="QA:$QA_EVAL,MT:$MT_EVAL" \
  OUT_JSON="$REPO/research_loop/state/diagnostics/DIAG-CONTENT_route_mass.json" \
  "$PY" "$RESDIR/measure_route_mass.py" >"$RESDIR/route_mass.log" 2>&1
  echo "DIAG_CONTENT_ROUTE_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$?"
fi

# =============================================================================
# solve-side stats: mean_mse, mass_on_S (solve-side), |v|max, timings
# =============================================================================
RUNDIR_A="$RD_A" RUNDIR_B="$RD_B" RUNDIR_C="$RD_C" \
OUT_JSON="$RESDIR/solve_stats.json" \
"$PY" "$RESDIR/collect_stats.py" >"$RESDIR/collect_stats.log" 2>&1
echo "DIAG_CONTENT_COLLECT_RC=$?"

echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
echo "DIAG_CONTENT_RC_B=$RC_B DIAG_CONTENT_RC_C=$RC_C"
echo "DIAG_CONTENT_DONE_TS=$(date -Is)"
cat "$SUMMARY"
