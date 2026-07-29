#!/usr/bin/env bash
# MECH-SEQUENTIAL (B-CASCADE / query-distribution): on-policy, layer-sequential
# reference-query re-extraction -- SCOUT-AM's divergence #3 / LIT-006. The one
# thing the AM paper does that this tree has never implemented.
#
#   arm control        : AM_ONPOLICY_LAYERS unset, KEY_MODE=freeze
#                        -> MUST reproduce MECH-KEYS control:
#                           in-run  QA 2.15320086479187 / MT 2.5296061038970947
#                           stand.  QA 2.159724712371826 / MT 2.529625177383423
#   arm onpolicy       : AM_ONPOLICY_LAYERS=4, KEY_MODE=freeze  (mechanism alone)
#   arm onpolicy_keys  : AM_ONPOLICY_LAYERS=4 + MECH-005 best point
#                        (KEY_MODE=highest_attention, AM_KEY_REPOSITION=1)
#
# GROUP SIZE = 4 (36 layers -> 9 groups, refreshes before layers 4,8,...,32 = 8
# extra reference-query passes per document). Per-layer would be 35 passes/doc;
# the brief's suggested range is 4-6 and 4 is the finer end, at an estimated
# +8-24 s/doc over the control's 11.4 s/doc -- i.e. a ~2-3x solve cost, well
# inside the precedent set by MECH-KEYS' OMP arm (21x).
#
# Everything else canonical top32 at theta=5e6, ENABLE_BETA=0 explicitly.
# All arms sequential on ONE flock-claimed GPU. PYTHONPATH pinned to _explore
# (RUNBOOK 6.10 / 9c-bis): unpinned, `import cartridges` resolves to the SIBLING
# repo, which has neither `rope_theta` nor `onpolicy_layers`.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
PY="$CARTRIDGES_DIR/.venv/bin/python"

RESDIR="$CARTRIDGES_DIR/research_loop/results/MECH-SEQUENTIAL"
mkdir -p "$RESDIR/evals"
exec > "$RESDIR/wrapper.log" 2>&1

echo "MS_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$CARTRIDGES_DIR" rev-parse HEAD)"

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 360); do
  for idx in 0 1; do
    lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
    exec {fd}>"$lockfile"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "MS_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then echo "MS_NO_FREE_GPU"; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MS_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"
# MANDATORY: prove the import path FROM /tmp (RUNBOOK 9c-bis -- repo root is a
# false negative because cwd is sys.path[0]).
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

# =====================================================================
# PART A: CUDA unit sanity (the CPU run already passed; see sanity_cpu.json)
# =====================================================================
echo "=== PART A: CUDA UNIT SANITY ==="
(cd /tmp && "$PY" "$RESDIR/sanity_seq.py" --cuda > "$RESDIR/sanity_cuda.json" 2> "$RESDIR/sanity_cuda.err") &
SPID=$!
for _ in $(seq 1 200); do
  "$PY" -c "import json,sys; json.load(open('$RESDIR/sanity_cuda.json'))" 2>/dev/null && break
  kill -0 "$SPID" 2>/dev/null || break
  sleep 5
done
# the CUDA interpreter can hang in nv_kthread_q_flush at exit AFTER writing its
# JSON (RUNBOOK 1/6.7) -> reap rather than wait.
kill -TERM "$SPID" 2>/dev/null; sleep 3; kill -KILL "$SPID" 2>/dev/null
"$PY" - <<'EOF'
import json
p = "/localhome/local-triv/gated-continual-cartridges_explore/research_loop/results/MECH-SEQUENTIAL/sanity_cuda.json"
try:
    d = json.load(open(p))
except Exception as e:
    print("SANITY_CUDA_UNREADABLE:", e); raise SystemExit(0)
c = d["arms"]["_checks"]
print("SANITY_CUDA import:", d["cartridges_import_path"])
print("SANITY_CUDA config:", json.dumps(d["config"]))
print("SANITY_CUDA checks:", json.dumps(c, indent=1))
print("SANITY_CUDA fail_loud:", json.dumps(d["fail_loud"]))
print("SANITY_CUDA off_vs_HEAD bit_identical:", d["off_vs_HEAD"].get("bit_identical"))
EOF

# =====================================================================
# PART B: the three arms
# =====================================================================
run_arm() {   # tag  key_mode  reposition  onpolicy  runname  log
  local tag="$1" kmode="$2" repos="$3" onp="$4" runname="$5" log="$6"
  echo "MS_${tag}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  export AM_ROPE_THETA=5000000
  export MAX_QUERIES_PER_HEAD=64
  export SLOT_SELECTION=tfidf
  if [ "$repos" = "-" ]; then unset AM_KEY_REPOSITION; else export AM_KEY_REPOSITION="$repos"; fi
  if [ "$onp" = "-" ]; then unset AM_ONPOLICY_LAYERS; else export AM_ONPOLICY_LAYERS="$onp"; fi
  USE_IDF=0 \
  GRANULARITY=per_layer \
  TOP_T=32 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE="$kmode" \
  ENABLE_BETA=0 \
  BETA_FIT_SCOPE=selected \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  RIDGE_LAMBDA_MIN=0.0 \
  DELTA_WEIGHT=1e-2 \
  AM_EXECUTION_MODE=per_document \
  SAVE_AFTER_EACH_DOCUMENT=1 \
  PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
  SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
  EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \
  RUN_NAME="$runname" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-CASCADE \
  WANDB_NOTES="on-policy layer-sequential re-extraction (AM paper divergence #3)" \
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  unset AM_KEY_REPOSITION AM_ROPE_THETA AM_ONPOLICY_LAYERS
  echo "MS_${tag}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  local dir
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  echo "MS_${tag}_RUNDIR=$dir"
  echo "MS_${tag}_WANDB=$(grep -oE 'https://wandb.ai/\S+/runs/[a-z0-9]+' "$log" | tail -1)"
  echo "MS_${tag}_INRUN=$(grep -E 'eval_(qa_forgetting|mt_acquisition)/loss ' "$log" | tr -s ' ' | tr '\n' ' ')"
  echo "MS_${tag}_ONPOLICY=$(grep -c 'on-policy refresh before layer' "$log") refresh lines"
  echo "MS_${tag}_ONPOLICY_FIRST=$(grep -m3 'on-policy refresh before layer' "$log" | tr '\n' '|')"
  [ -n "$dir" ] && echo "MS_${tag}_SUMMARY=$(cat "$dir/phase2_summary.json" 2>/dev/null | tr -d '\n ')"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

echo "=== PART B: ARMS ==="
run_arm control       freeze            -  -  MECH-SEQUENTIAL_control       "$RESDIR/run_control.log";       RC_C=$?
run_arm onpolicy      freeze            -  4  MECH-SEQUENTIAL_onpolicy      "$RESDIR/run_onpolicy.log";      RC_O=$?
run_arm onpolicy_keys highest_attention 1  4  MECH-SEQUENTIAL_onpolicy-keys "$RESDIR/run_onpolicy_keys.log"; RC_K=$?

DIR_C=$(cat "$RESDIR/rundir_control.txt" 2>/dev/null)
DIR_O=$(cat "$RESDIR/rundir_onpolicy.txt" 2>/dev/null)
DIR_K=$(cat "$RESDIR/rundir_onpolicy_keys.txt" 2>/dev/null)
echo "MS_RC control=$RC_C onpolicy=$RC_O onpolicy_keys=$RC_K"
echo "DIR_control=$DIR_C"
echo "DIR_onpolicy=$DIR_O"
echo "DIR_onpolicy_keys=$DIR_K"

# ---------------------------------------------------------------- independent evals
eval_one() {  # label ckpt split [tag]
  local label="$1" ckpt="$2" split="$3" tag="${4:-}"
  local log="$RESDIR/evals/eval_${label}${tag}_${split}.log"
  : > "$log"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="MECH-SEQUENTIAL_${label}${tag}_eval${split}" \
  WANDB_GROUP=B-CASCADE \
  WANDB_NOTES="on-policy layer-sequential re-extraction (AM paper divergence #3)" \
  WANDB_DISABLED=0 \
  "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 240); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  local loss url
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/\S+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  echo "RESULT ${label}${tag} ${split}: loss=${loss:-MISSING} url=${url:-MISSING} ckpt=$ckpt"
  printf "%s\t%s\t%s\t%s\t%s\n" "$label" "${tag:-k16}" "$split" "${loss:-MISSING}" "${url:-MISSING}" >> "$RESDIR/curve.tsv"
  # RUNBOOK 1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

printf "arm\tk\tsplit\tloss\twandb_url\n" > "$RESDIR/curve.tsv"

echo "=== INDEPENDENT EVALS (k=16 endpoints) ==="
for SPLIT in QA MT; do
  [ -n "$DIR_C" ] && eval_one control       "$DIR_C/cache_last.pt" "$SPLIT"
  [ -n "$DIR_O" ] && eval_one onpolicy      "$DIR_O/cache_last.pt" "$SPLIT"
  [ -n "$DIR_K" ] && eval_one onpolicy_keys "$DIR_K/cache_last.pt" "$SPLIT"
done

# ---- k-curve shape: the value-only curve bottoms at k=12 then DEGRADES by 0.117.
# Does on-policy change that shape? (k=16 already measured above.)
ckpt_for() {  # run_dir k -> cache-after-doc-<k-1>-*.pt
  ls "$1"/cache-after-doc-$(printf "%03d" $(($2-1)))-*.pt 2>/dev/null | head -1
}
echo "=== K-CURVE (k=8,12) ==="
for K in 12 8; do
  for A in "onpolicy:$DIR_O" "onpolicy_keys:$DIR_K"; do
    L="${A%%:*}"; D="${A#*:}"
    [ -z "$D" ] && continue
    C=$(ckpt_for "$D" "$K")
    if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=$L k=$K"; continue; fi
    eval_one "$L" "$C" QA "_k$K"
    eval_one "$L" "$C" MT "_k$K"
  done
done

# --------------------------------------------- eval-time mass_on_S + cartridge mass
echo "=== ROUTE-MASS DIAGNOSTIC ==="
PHASE1="$CARTRIDGES_DIR/outputs/phase1_selfdistill_qwen512/cache_last.pt"
SPECS=""
[ -n "$DIR_C" ] && SPECS="${SPECS}control:$DIR_C/cache_last.pt:$DIR_C,"
[ -n "$DIR_O" ] && SPECS="${SPECS}onpolicy:$DIR_O/cache_last.pt:$DIR_O,"
[ -n "$DIR_K" ] && SPECS="${SPECS}onpolicy_keys:$DIR_K/cache_last.pt:$DIR_K,"
SPECS="${SPECS}phase1:$PHASE1:$DIR_C"
echo "CACHE_SPECS=$SPECS"
CACHE_SPECS="$SPECS" \
EVAL_SPECS="QA:data/qasper/eval/qasper_eval_QA.parquet,MT:data/qasper/eval/qasper_eval_MT.parquet" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-SEQUENTIAL_route_mass.json" \
"$PY" "$RESDIR/measure_route_mass.py" > "$RESDIR/route_mass.log" 2>&1
echo "ROUTE_MASS_RC=$?"
tail -40 "$RESDIR/route_mass.log"

# --------------------------------------------------------------- per-layer collection
echo "=== COLLECT PER-DOC / PER-LAYER STATS ==="
ARMS="control:$DIR_C,onpolicy:$DIR_O,onpolicy_keys:$DIR_K" \
CONTROL_ARM=control \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-SEQUENTIAL.json" \
"$PY" "$RESDIR/collect_stats.py"
echo "COLLECT_RC=$?"

echo "=== CURVE ==="
cat "$RESDIR/curve.tsv"
echo "MS_ALL_DONE_TS=$(date -Is)"
exit $(( RC_C != 0 || RC_O != 0 || RC_K != 0 ))
