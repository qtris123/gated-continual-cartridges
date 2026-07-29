#!/usr/bin/env bash
# DIAG-PERDOC (B-OVERWRITE / capacity): the PER-DOCUMENT acquisition ceiling.
#
# Question: does writing ONE document alone into 32 slots (no competition) teach the
# model that document -- and by how much more than writing it as one of 16?
#
# NO source file is edited (MECH-KEYS is editing cartridges/am/key_select.py).
# Imports + driver are pinned to a FROZEN SNAPSHOT of committed code (RUNBOOK 9c-bis).
#
# Two arms, each internally config-consistent:
#   A  AM_ROPE_THETA unset (= 10000, the historical default): matches EXACTLY the
#      DIAG-OVERWRITE canonical run whose cache-after-doc-*.pt snapshots supply the
#      k=12 / k=16 baselines and the Phase-1 floor already on the board.
#   B  AM_ROPE_THETA=5000000 (the brief's canonical): baselines taken from DIAG-ROPE
#      arm B's OWN per-document snapshots, so the comparison stays within-config.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_perdoc
DATA=/tmp/perdoc
RESDIR="$REPO/research_loop/results/DIAG-PERDOC"
LOGDIR="$RESDIR/logs"
mkdir -p "$LOGDIR"

export PATH="$REPO/.venv/bin:$PATH"            # so the wrapper's bare `python` is OUR venv
export CARTRIDGES_DIR="$SNAP"                  # driver + package both from HEAD
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
PY="$REPO/.venv/bin/python"

exec > "$RESDIR/wrapper.log" 2>&1
echo "PERDOC_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
echo "SNAPSHOT_PATH=$SNAP"
echo "SNAPSHOT_MANIFEST=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"

# ---- claim exactly one GPU via flock -----------------------------------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 120); do
  for idx in 0 1; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "PERDOC_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 30
done
if [ -z "$CLAIMED" ]; then echo "PERDOC_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "PERDOC_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT"

# ---- MANDATORY: prove the pin, from /tmp (NOT the repo root) -----------------
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

PHASE1="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
OVW_RUN="$REPO/outputs/2026-07-28-21-43-17-continual_am_sparse/4b0edcab-d1c5-4313-bc43-309d7ff77097"      # theta=1e4 canonical 16-doc
ROPEB_RUN="$REPO/outputs/2026-07-28-21-23-35-continual_am_sparse/5ff66d3f-84dc-466e-b539-d2ef01a53785"    # theta=5e6 16-doc (DIAG-ROPE arm B)

WRITES="$RESDIR/writes.tsv"
EVALS="$RESDIR/evals.tsv"
[ -s "$WRITES" ] || printf "tag\tdoc_index\trope\tsynth\trun_dir\tcache\tsecs\n" > "$WRITES"
[ -s "$EVALS" ]  || printf "ckpt_tag\tsubset\tloss\tckpt\twandb_url\n" > "$EVALS"

# ------------------------------------------------------------------ one write --
do_write() {   # tag doc_index rope synth_parquet
  local tag="$1" idx="$2" rope="$3" synth="$4"
  grep -qP "^${tag}\t" "$WRITES" && { echo "SKIP_WRITE $tag (already done)"; return 0; }
  local log="$LOGDIR/write_${tag}.log"
  local t0=$(date +%s)
  echo "--- [$(date -Is)] WRITE $tag idx=$idx rope=$rope synth=$synth"
  local ropeenv=""
  if [ "$rope" != "default" ]; then ropeenv="$rope"; fi
  (
    if [ -n "$ropeenv" ]; then export AM_ROPE_THETA="$ropeenv"; fi
    USE_IDF=0 \
    SLOT_SELECTION=tfidf \
    GRANULARITY=per_layer \
    TOP_T=32 \
    TARGET_MODE=cartridge_plus_doc \
    KEY_MODE=freeze \
    ENABLE_BETA=0 \
    RIDGE_LAMBDA=1e-4 \
    RIDGE_SCALE=spectral \
    DELTA_WEIGHT=1e-2 \
    MAX_QUERIES_PER_HEAD=64 \
    MAX_REF_EXAMPLES_PER_DOC=32 \
    AM_EXECUTION_MODE=per_document \
    SAVE_AFTER_EACH_DOCUMENT=1 \
    PHASE1_CACHE_PATH="$PHASE1" \
    SYNTH_DATA_PATH="$synth" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    NUM_TOKENS=512 \
    RUN_NAME="DIAG-PERDOC_${tag}_write" \
    WANDB_DISABLED=0 \
    WANDB_GROUP=B-OVERWRITE \
    WANDB_NOTES="single-document AM write (doc_index=$idx, rope=$rope) vs the 16-document canonical cartridge" \
    bash "$SNAP/examples/qasper2/scripts/core/train_continual_am_sparse.sh"
  ) >"$log" 2>&1
  local rc=$? t1=$(date +%s)
  local run_dir cache
  # Identify the run by the artefact THIS write created (mtime newer than launch),
  # never by "newest directory" -- another worker may be writing concurrently.
  cache=$(find "$REPO/outputs" -maxdepth 3 -name 'cache-after-doc-000-*.pt' \
            -newermt "@$((t0-1))" -print 2>/dev/null | sort | tail -1)
  run_dir=$(dirname "$cache" 2>/dev/null)
  echo "WRITE_DONE $tag rc=$rc secs=$((t1-t0)) run_dir=$run_dir cache=$cache"
  if [ -z "$cache" ]; then echo "WRITE_FAILED_NO_CACHE $tag"; tail -25 "$log"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$idx" "$rope" "$synth" "$run_dir" "$cache" "$((t1-t0))" >> "$WRITES"
}

# ------------------------------------------------------------------- one eval --
eval_one() {   # ckpt_tag subset_name ckpt_path eval_parquet
  local ctag="$1" sub="$2" ckpt="$3" data="$4"
  grep -qP "^${ctag}\t${sub}\t" "$EVALS" && { echo "SKIP_EVAL $ctag/$sub"; return 0; }
  if [ ! -f "$ckpt" ]; then echo "MISSING_CKPT $ctag $ckpt"; return 1; fi
  local log="$LOGDIR/eval_${ctag}__${sub}.log"
  : > "$log"
  echo "--- [$(date -Is)] EVAL ckpt=$ctag subset=$sub"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="$data" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="DIAG-PERDOC_${ctag}_${sub}" \
  WANDB_GROUP=B-OVERWRITE \
  WANDB_DISABLED=0 \
  "$PY" "$SNAP/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 200); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  local loss url
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  echo "RESULT $ctag $sub: loss=${loss:-MISSING} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\n" "$ctag" "$sub" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" >> "$EVALS"
  # RUNBOOK 1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 3; kill -KILL "$pid" 2>/dev/null
  sleep 1
}

DOCS="000 009 011 013 015"

# =============================== PHASE 1: the single-document writes ===========
for D in $DOCS; do
  do_write "A_doc${D}" "$((10#$D))" default "$DATA/synth/doc${D}.parquet"
done
do_write "A_doc015shuf" 15 default "$DATA/synth/doc015_shuf.parquet"
for D in $DOCS; do
  do_write "B_doc${D}" "$((10#$D))" 5000000 "$DATA/synth/doc${D}.parquet"
done
echo "PERDOC_WRITES_DONE_TS=$(date -Is)"
cat "$WRITES"

cache_of() {   # tag -> cache path
  grep -P "^$1\t" "$WRITES" | tail -1 | cut -f6
}

# =============================== PHASE 2: the eval matrix =====================
MTFULL="$REPO/data/qasper/eval/qasper_eval_MT.parquet"

# (a) baselines shared by both arms: Phase-1 floor on every per-document subset
for D in $DOCS; do
  eval_one PH1 "doc${D}" "$PHASE1" "$DATA/eval/mt_doc${D}.parquet"
done
# decomposition check: the union of the five subsets, same checkpoint
eval_one PH1 union5 "$PHASE1" "$DATA/eval/mt_union5.parquet"

# (b) arm A: 16-document and k=12 caches from the canonical theta=1e4 run
K16A=$(ls "$OVW_RUN"/cache-after-doc-015-*.pt | head -1)
K12A=$(ls "$OVW_RUN"/cache-after-doc-011-*.pt | head -1)
for D in $DOCS; do
  eval_one K16A "doc${D}" "$K16A" "$DATA/eval/mt_doc${D}.parquet"
  eval_one K12A "doc${D}" "$K12A" "$DATA/eval/mt_doc${D}.parquet"
done

# (c) arm A: each single-document cache on its OWN subset and on the full MT set
for D in $DOCS; do
  C=$(cache_of "A_doc${D}")
  eval_one "A_doc${D}" "doc${D}" "$C" "$DATA/eval/mt_doc${D}.parquet"
  eval_one "A_doc${D}" "MTfull" "$C" "$MTFULL"
done
C=$(cache_of A_doc015shuf)
eval_one A_doc015shuf doc015 "$C" "$DATA/eval/mt_doc015.parquet"
eval_one A_doc015shuf MTfull "$C" "$MTFULL"

# (d) arm B: within-config baselines + the single-document caches
K16B=$(ls "$ROPEB_RUN"/cache-after-doc-015-*.pt | head -1)
K12B=$(ls "$ROPEB_RUN"/cache-after-doc-011-*.pt | head -1)
for D in $DOCS; do
  eval_one K16B "doc${D}" "$K16B" "$DATA/eval/mt_doc${D}.parquet"
  eval_one K12B "doc${D}" "$K12B" "$DATA/eval/mt_doc${D}.parquet"
  C=$(cache_of "B_doc${D}")
  eval_one "B_doc${D}" "doc${D}" "$C" "$DATA/eval/mt_doc${D}.parquet"
  eval_one "B_doc${D}" "MTfull" "$C" "$MTFULL"
done

echo "PERDOC_EVALS_DONE_TS=$(date -Is)"
echo "SNAPSHOT_MANIFEST_AFTER=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
cat "$EVALS"
echo "PERDOC_ALL_DONE_TS=$(date -Is)"
