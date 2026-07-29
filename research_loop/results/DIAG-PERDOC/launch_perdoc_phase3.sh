#!/usr/bin/env bash
# DIAG-PERDOC phase 3: correctness gates + the leave-one-out uncertainty probe.
#
#  (a) GATES: the three reference checkpoints on the FULL MT set, in this session, with
#      this harness -- must reproduce Phase-1 3.7825 / k=12 2.4352 / k=16 2.5524.
#  (b) LOO: doc-015's 7 eval questions, each held out in turn, under the solo cache and
#      the 16-document cache. Because the harness loss is a token-weighted mean over
#      scored tokens and per-example CE is packing-invariant, the per-example CE is
#      recovered exactly:  c_e = L(S)*T_S - L(S\e)*T_{S\e}.  That gives a real
#      per-example spread / paired-delta standard error for a 7-question subset.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_perdoc
DATA=/tmp/perdoc
RESDIR="$REPO/research_loop/results/DIAG-PERDOC"
LOGDIR="$RESDIR/logs"
mkdir -p "$LOGDIR"

export PATH="$REPO/.venv/bin:$PATH"
export CARTRIDGES_DIR="$SNAP"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
PY="$REPO/.venv/bin/python"

exec > "$RESDIR/wrapper_phase3.log" 2>&1
echo "PERDOC3_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
echo "SNAPSHOT_MANIFEST=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 120); do
  for idx in 0 1; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "PERDOC3_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 30
done
if [ -z "$CLAIMED" ]; then echo "PERDOC3_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
echo "PERDOC3_CLAIMED_GPU=$CLAIMED"
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

PHASE1="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
OVW_RUN="$REPO/outputs/2026-07-28-21-43-17-continual_am_sparse/4b0edcab-d1c5-4313-bc43-309d7ff77097"
ROPEB_RUN="$REPO/outputs/2026-07-28-21-23-35-continual_am_sparse/5ff66d3f-84dc-466e-b539-d2ef01a53785"
MTFULL="$REPO/data/qasper/eval/qasper_eval_MT.parquet"
EVALS="$RESDIR/evals.tsv"
WRITES="$RESDIR/writes.tsv"

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
  kill -TERM "$pid" 2>/dev/null; sleep 3; kill -KILL "$pid" 2>/dev/null
  sleep 1
}

cache_of() { grep -P "^$1\t" "$WRITES" | tail -1 | cut -f6; }

K16A=$(ls "$OVW_RUN"/cache-after-doc-015-*.pt | head -1)
K12A=$(ls "$OVW_RUN"/cache-after-doc-011-*.pt | head -1)
K16B=$(ls "$ROPEB_RUN"/cache-after-doc-015-*.pt | head -1)
K12B=$(ls "$ROPEB_RUN"/cache-after-doc-011-*.pt | head -1)
K01B=$(ls "$ROPEB_RUN"/cache-after-doc-000-*.pt | head -1)

# (a) gates on the full MT set
eval_one PH1  MTfull "$PHASE1" "$MTFULL"
eval_one K16A MTfull "$K16A"   "$MTFULL"
eval_one K12A MTfull "$K12A"   "$MTFULL"
eval_one K16B MTfull "$K16B"   "$MTFULL"
eval_one K12B MTfull "$K12B"   "$MTFULL"
# k=1 of the theta=5e6 16-document run: doc_index 0 gets seed 0 in BOTH routes, so this
# must reproduce the B_doc000 solo write exactly -- the gate for the whole theta=5e6 arm.
eval_one K01B MTfull "$K01B"   "$MTFULL"
eval_one K01B doc000 "$K01B"   "$DATA/eval/mt_doc000.parquet"

# (b) leave-one-out on doc-015's own 7 questions
A15=$(cache_of A_doc015)
for i in 0 1 2 3 4 5 6; do
  eval_one A_doc015 "loo_e${i}" "$A15"  "$DATA/eval/loo_doc015_e${i}.parquet"
  eval_one K16A     "loo_e${i}" "$K16A" "$DATA/eval/loo_doc015_e${i}.parquet"
done

echo "PERDOC3_DONE_TS=$(date -Is)"
echo "SNAPSHOT_MANIFEST_AFTER=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
