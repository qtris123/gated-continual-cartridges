#!/usr/bin/env bash
# VERIFY-OPTIMA (W5 VERIFY, GPU) — seed-vary the OWN-OPTIMUM comparison.
#
# MECH-SEED seed-varied the MATCHED-k comparison only (k=12, k=16). DIAG-CONTROLCURVE
# relocated the frozen control's true MT minimum to k=10. So the fair comparison
# (keys at its argmin k=12 vs control at its argmin k=10) has never been seed-varied.
# This job evaluates EXISTING per-document snapshots. NOTHING IS RE-TRAINED.
#
# Snapshot mapping: SAVE_AFTER_EACH_DOCUMENT=1 -> k documents written == cache-after-doc-(k-1).
#
# Import discipline (RUNBOOK §9c-bis): DIAG-IMPORTANCE may be editing source, so every
# eval runs against a frozen `git archive HEAD` snapshot, from cwd=/tmp/..., and each eval
# log prints the resolved cartridges package path (runpy wrapper, no source edit).
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_verifyoptima
WORKCWD=/tmp/verify_optima_cwd
PY="$REPO/.venv/bin/python"
RESDIR="$REPO/research_loop/results/VERIFY-OPTIMA"
LOGDIR="$RESDIR/evals"
mkdir -p "$LOGDIR" "$WORKCWD"
exec > "$RESDIR/wrapper_addendum.log" 2>&1

export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0

echo "VO_ADDENDUM_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain > "$RESDIR/git_status_at_launch.txt" 2>&1
cat "$RESDIR/git_status_at_launch.txt"

# ---- frozen HEAD snapshot -----------------------------------------------------
rm -rf "$SNAP"; mkdir -p "$SNAP"
git -C "$REPO" archive HEAD cartridges examples | tar -x -C "$SNAP"
ln -sfn "$REPO/.venv" "$SNAP/.venv"
ln -sfn "$REPO/data"  "$SNAP/data"
manifest() { (cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort \
  | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1); }
echo "SNAPSHOT_PATH=$SNAP"
echo "SNAPSHOT_MANIFEST_AT_LAUNCH=$(manifest)"
export CARTRIDGES_DIR="$SNAP"
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
cd "$WORKCWD" || exit 9
echo "PIN_PROBE_CWD=$PWD"
echo "PIN_PROBE=$("$PY" -c 'import cartridges,os;print(os.path.dirname(cartridges.__file__))')"

# ---- claim exactly one GPU via flock (never steal) ---------------------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 480); do
  for idx in 1 0; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "VO_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then echo "VO_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "VO_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

SUMMARY="$RESDIR/curve.tsv"   # APPEND: addendum run, header already written

# run dirs (all pre-existing; nothing re-trained) ------------------------------
D_KEYS_0="$REPO/outputs/2026-07-29-04-13-05-continual_am_sparse/955753b0-2246-4f18-8200-729b831687e5"
D_KEYS_1000="$REPO/outputs/2026-07-29-04-40-40-continual_am_sparse/92b047bb-8822-4f50-8720-5dc2fcb26a4f"
D_KEYS_2000="$REPO/outputs/2026-07-29-05-06-27-continual_am_sparse/5d1785dc-3bc8-40d4-b177-12b7a20f1743"
D_CTRL_0="$REPO/outputs/2026-07-28-23-19-56-continual_am_sparse/508b8b0f-075a-429b-b757-2b11bfb57124"
D_CTRL_1000="$REPO/outputs/2026-07-29-05-37-31-continual_am_sparse/a2f9458c-31a0-4bfb-8b60-da98da202ba6"
D_CTRL_2000="$REPO/outputs/2026-07-29-06-02-01-continual_am_sparse/727e44ac-de50-44df-bf3c-877ad3e65317"

dir_for() {
  case "$1_$2" in
    keys_0) echo "$D_KEYS_0";; keys_1000) echo "$D_KEYS_1000";; keys_2000) echo "$D_KEYS_2000";;
    control_0) echo "$D_CTRL_0";; control_1000) echo "$D_CTRL_1000";; control_2000) echo "$D_CTRL_2000";;
    *) echo "";;
  esac
}
ckpt_for() { ls "$1"/cache-after-doc-$(printf "%03d" $(($2-1)))-*.pt 2>/dev/null | head -1; }

eval_one() {  # arm offset k split
  local arm="$1" off="$2" k="$3" split="$4"
  local dir ckpt
  dir=$(dir_for "$arm" "$off")
  if [ -z "$dir" ] || [ ! -d "$dir" ]; then echo "MISSING_RUNDIR arm=$arm off=$off"; return 1; fi
  ckpt=$(ckpt_for "$dir" "$k")
  if [ -z "$ckpt" ]; then
    echo "MISSING_SNAPSHOT arm=${arm} off=${off} k=${k} dir=${dir} (expected cache-after-doc-$(printf '%03d' $((k-1)))-*.pt)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$off" "$k" "$split" "MISSING_SNAPSHOT" "-" "-" "0" "-" >> "$SUMMARY"
    return 1
  fi
  local log="$LOGDIR/eval_${arm}_off${off}_k${k}_${split}.log"
  : > "$log"
  echo "--- [$(date -Is)] EVAL arm=${arm} off=${off} k=${k} split=${split} ckpt=${ckpt} sha=$(sha256sum "$ckpt" | cut -c1-16)"
  (
    cd "$WORKCWD" || exit 9
    CHECKPOINT_PATH="$ckpt" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    EVAL_MODE=cartridge \
    RUN_NAME="VERIFY-OPTIMA_${arm}_off${off}_k${k}_${split}" \
    WANDB_GROUP=VERIFY \
    WANDB_NOTES="own-optimum comparison seed-varied: control k=10 vs keys k=12 (VERIFY-OPTIMA)" \
    WANDB_DISABLED=0 \
    "$PY" -c "
import os, sys, runpy, cartridges
print('CARTRIDGES_PKG ' + os.path.dirname(cartridges.__file__), flush=True)
sys.argv = [os.environ['VO_SCRIPT']]
runpy.run_path(os.environ['VO_SCRIPT'], run_name='__main__')
" >"$log" 2>&1
  ) &
  local pid=$!
  local i
  for i in $(seq 1 400); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  local loss url ckptok pkg
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  ckptok=$(grep -c -F -- "$ckpt" "$log")
  pkg=$(grep -oE '^CARTRIDGES_PKG .*' "$log" | head -1 | awk '{print $2}')
  echo "RESULT arm=${arm} off=${off} k=${k} ${split}: loss=${loss:-MISSING} ckpt_in_log=${ckptok} pkg=${pkg:-MISSING} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$off" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" "$ckptok" "${pkg:-MISSING}" >> "$SUMMARY"
  # RUNBOOK §1: eval_forgetting.py HANGS after printing `Eval loss` -> reap the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

export VO_SCRIPT="$SNAP/examples/qasper2/train/eval_forgetting.py"
echo "VO_SCRIPT=$VO_SCRIPT"

# =============================================================================
# BLOCK 4 — ARGMIN STABILITY: does the control's MT argmin stay at k=10 under a
#           changed seed?  MT only, both non-zero offsets, neighbourhood first.
# =============================================================================
echo "############ BLOCK 4: CONTROL MT ARGMIN STABILITY ############"
for K in 3 2 1; do
  eval_one control 1000 "$K" MT
  eval_one control 2000 "$K" MT
done

echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain > "$RESDIR/git_status_after.txt" 2>&1
echo "VO_ADDENDUM_DONE_TS=$(date -Is)"
cat "$SUMMARY"
