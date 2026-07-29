#!/usr/bin/env bash
# DIAG-CONTROLCURVE (B-ROUTE, key side) -- W5 VERIFY
#
# DIAG-KEYCURVE mapped the keys+reposition arm at all 16 k but evaluated the
# frozen-key CONTROL at only k in {8,12,14,16}. The control's lowest measured MT
# (2.4536) sits at k=8, the EDGE of that set, so its true minimum is UNLOCATED and
# the headline "keys beat frozen keys" compares a mapped curve against an unmapped
# one. This job locates the control's true minimum.
#
# Control arm = MECH-KEYS `control`: KEY_MODE=freeze, AM_ROPE_THETA=5e6, ENABLE_BETA=0,
# top32 per_layer tfidf. NOTHING IS RE-TRAINED -- the 16 `cache-after-doc-*.pt`
# snapshots already exist in that run dir.
#
# PART 0  correctness gates (fail-fast): control k=16 must give
#         QA 2.159724712371826 / MT 2.529625177383423 and Phase-1 (k=0) must give
#         QA 2.23880672454834 / MT 3.7825491428375244.
# PART 1  the control at EVERY k = 1..16 on BOTH splits (the brief only requires the
#         13 k DIAG-KEYCURVE skipped; k=8/12/14 are re-run anyway as 3 extra
#         same-session reproduction checks, and both numbers are reported).
# PART 2  eval-time mass_on_S / MT-QA mass ratio / total cartridge mass at every k.
#
# Imports pinned to a frozen `git archive HEAD` snapshot (RUNBOOK 9c-bis): a
# concurrent worker (MECH-SEQUENTIAL) is editing cartridges/am/finetune.py,
# cartridges/am/continual.py and examples/qasper2/train/continual_am_sparse.py.
# NO SOURCE FILE IS EDITED BY THIS JOB.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_ctrl
PY="$REPO/.venv/bin/python"

export CARTRIDGES_DIR="$SNAP"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"

RESDIR="$REPO/research_loop/results/DIAG-CONTROLCURVE"
LOGDIR="$RESDIR/evals"
mkdir -p "$LOGDIR"
exec > "$RESDIR/wrapper.log" 2>&1

cd "$REPO" || exit 9

# the pinned snapshot needs a venv + data under CARTRIDGES_DIR (git archive ships neither)
ln -sfn "$REPO/.venv" "$SNAP/.venv"
ln -sfn "$REPO/data"  "$SNAP/data"

manifest() { (cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort \
  | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1); }

echo "CC_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
echo "SNAPSHOT_PATH=$SNAP"
echo "SNAPSHOT_MANIFEST_AT_LAUNCH=$(manifest)"

# ---- claim exactly one GPU via flock (never steal) ---------------------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 240); do
  for idx in 0 1; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "CC_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then echo "CC_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "CC_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

# ---- MANDATORY: prove the pin, from /tmp (NOT the repo root) -----------------
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

DIR_C="$REPO/outputs/2026-07-28-23-19-56-continual_am_sparse/508b8b0f-075a-429b-b757-2b11bfb57124"  # MECH-KEYS control (KEY_MODE=freeze)
PHASE1="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
echo "CC_DIR_control=$DIR_C"
echo "CC_PHASE1=$PHASE1"

# snapshot identity: all 16 must be pairwise distinct files
echo "--- control snapshot sha256 ---"
sha256sum "$DIR_C"/cache-after-doc-*.pt
echo "--- end sha256 ---"

SUMMARY="$RESDIR/curve.tsv"
: > "$SUMMARY"
printf "arm\tk\tsplit\tloss\tckpt\twandb_url\n" >> "$SUMMARY"

eval_one() {   # arm k split ckpt
  local arm="$1" k="$2" split="$3" ckpt="$4"
  local log="$LOGDIR/eval_${arm}_doc${k}_${split}.log"
  : > "$log"
  echo "--- [$(date -Is)] arm=${arm} k=${k} split=${split} ckpt=${ckpt}"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="DIAG-CONTROLCURVE_doc${k}_${split}" \
  WANDB_GROUP=B-ROUTE \
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

ckpt_for() {  # run_dir k  -> the cache-after-doc-<k-1>-*.pt path
  local dir="$1" k="$2"
  ls "$dir"/cache-after-doc-$(printf "%03d" $((k-1)))-*.pt 2>/dev/null | head -1
}

# =====================================================================
# PART 0 -- CORRECTNESS GATES (fail fast)
# =====================================================================
echo "=== PART 0: correctness gates ==="
C16=$(ckpt_for "$DIR_C" 16)
eval_one control 16 QA "$C16"
eval_one control 16 MT "$C16"
eval_one phase1 0 QA "$PHASE1"
eval_one phase1 0 MT "$PHASE1"

GATE=$("$PY" - "$SUMMARY" <<'EOF'
import sys
exp = {("control","16","QA"): "2.159724712371826",
       ("control","16","MT"): "2.529625177383423",
       ("phase1","0","QA"):   "2.23880672454834",
       ("phase1","0","MT"):   "3.7825491428375244"}
got = {}
for line in open(sys.argv[1]).read().splitlines()[1:]:
    a, k, sp, loss = line.split("\t")[:4]
    got[(a, k, sp)] = loss
ok = True
for key, want in exp.items():
    have = got.get(key, "MISSING")
    same = (have == want)
    ok &= same
    print(f"GATE {key} expected={want} observed={have} match={same}", file=sys.stderr)
print("PASS" if ok else "FAIL")
EOF
)
echo "CC_GATE=$GATE"
if [ "$GATE" != "PASS" ]; then
  echo "CC_ABORT gate failed -- stopping per brief" >&2
  echo "CC_DONE_TS=$(date -Is)"
  echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
  exit 4
fi

# =====================================================================
# PART 1 -- the control's full k curve, both splits.
#   Brief-required (never measured before): k = 1..7, 9, 10, 11, 13, 15.
#   k = 8, 12, 14 are re-run too, as same-session reproduction checks against
#   DIAG-KEYCURVE (2.453586/1.932971, 2.470458/2.074141, 2.483512/2.092168).
# =====================================================================
echo "=== PART 1: control freeze k-curve ==="
for K in 4 2 6 1 3 5 7 9 10 11 13 15 8 12 14; do
  C=$(ckpt_for "$DIR_C" "$K")
  if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=control k=$K"; continue; fi
  eval_one control "$K" QA "$C"
  eval_one control "$K" MT "$C"
done

# =====================================================================
# PART 2 -- eval-time mass_on_S / MT-QA ratio / cartridge mass at every k.
#   Per-k slot set S = union of the first k documents' per-layer top-32 selections.
# =====================================================================
echo "=== PART 2: route-mass per k ==="
SLOTDIR=/tmp/ccurve_slots
rm -rf "$SLOTDIR"; mkdir -p "$SLOTDIR"
build_slotdir() {  # run_dir k label -> echoes the symlink dir
  local dir="$1" k="$2" label="$3"
  local d="$SLOTDIR/${label}_k${k}"
  mkdir -p "$d"
  local i
  for i in $(seq 0 $((k-1))); do
    ln -sf "$dir"/am_doc_doc-$(printf "%03d" "$i")-*.pt "$d"/ 2>/dev/null
  done
  echo "$d"
}
SPECS=""
for K in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  C=$(ckpt_for "$DIR_C" "$K"); [ -z "$C" ] && continue
  D=$(build_slotdir "$DIR_C" "$K" control)
  SPECS="${SPECS}control_k${K}:$C:$D,"
done
# Phase-1 reference measured on the CONTROL's full 16-document union, exactly as
# MECH-KEYS/DIAG-KEYCURVE did -> its 0.08778 (MT) / 0.08119 (QA) is a cross-session check.
SPECS="${SPECS}phase1:$PHASE1:$DIR_C"
echo "CC_CACHE_SPECS=$SPECS"
cp "$REPO/research_loop/results/DIAG-KEYCURVE/measure_route_mass.py" "$RESDIR/measure_route_mass.py"
echo "CC_ROUTEMASS_SCRIPT_SHA256=$(sha256sum "$RESDIR/measure_route_mass.py" | cut -d' ' -f1)"
CACHE_SPECS="$SPECS" \
EVAL_SPECS="QA:$REPO/data/qasper/eval/qasper_eval_QA.parquet,MT:$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
OUT_JSON="$REPO/research_loop/state/diagnostics/DIAG-CONTROLCURVE_route_mass.json" \
"$PY" "$RESDIR/measure_route_mass.py" > "$RESDIR/route_mass.log" 2>&1
echo "CC_ROUTE_MASS_RC=$?"
tail -45 "$RESDIR/route_mass.log"

echo "CC_DONE_TS=$(date -Is)"
echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
cat "$SUMMARY"
