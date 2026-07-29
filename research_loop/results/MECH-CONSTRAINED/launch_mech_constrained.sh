#!/usr/bin/env bash
# MECH-CONSTRAINED (W3 BUILD + W4 TEST) — B-GATE / MECH-009.
#
# The one variant of the "protect Phase-1, still acquire" idea that was never
# run: best-t MASS-RANKED **WITHIN** a hard safety constraint (the safest `q` of
# slots per layer by redundancy), rather than ranked BY the safety metric.
# MECH-INFOGATE tested the latter and diagnosed that DIAG-IMPORTANCE's promising
# projection described the former. This traces the constraint-strength curve.
#
# Arms (all else at the project's current best point, MECH-005 config):
#   C0    tfidf                                TOP_T=32   <- GATE (must reproduce)
#   Q75   constrained_mass q=0.75 redundancy   TOP_T=32
#   Q50   constrained_mass q=0.50 redundancy   TOP_T=32
#   Q25   constrained_mass q=0.25 redundancy   TOP_T=32   (DIAG-IMPORTANCE's quartile)
#   Q50T64 constrained_mass q=0.50 redundancy  TOP_T=64   (loosest x least-bad budget)
#
# Every arm evaluated on BOTH splits at k = 8, 10, 12, 16.
#
# No import pin: this worker IS the source editor and the runs must see its edit.
# PYTHONPATH is pinned to the _explore repo so `import cartridges` cannot land on
# the sibling package (RUNBOOK §6.10 / §9c-bis).
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
PY="$REPO/.venv/bin/python"
RESDIR="$REPO/research_loop/results/MECH-CONSTRAINED"
LOGDIR="$RESDIR/evals"; RUNLOG="$RESDIR/runs"
mkdir -p "$LOGDIR" "$RUNLOG"
exec > "$RESDIR/wrapper.log" 2>&1

cd "$REPO" || exit 9
export CARTRIDGES_DIR="$REPO"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

NOTES="mass-ranked within a safety constraint — the variant DIAG-IMPORTANCE projected"

echo "MC_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" diff --stat -- cartridges examples > "$RESDIR/worktree_diff_stat.txt" 2>&1
git -C "$REPO" diff -- cartridges examples > "$RESDIR/worktree_diff.patch" 2>&1
echo "WORKTREE_DIFF_STAT:"; cat "$RESDIR/worktree_diff_stat.txt"
echo "IMPORT_PROBE_FROM_TMP=$(cd /tmp && "$PY" -c 'import cartridges,os;print(os.path.dirname(cartridges.__file__))')"

# ---- claim exactly one GPU via flock (never steal) ---------------------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 480); do
  for idx in 0 1; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "MC_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
[ -z "$CLAIMED" ] && { echo "MC_NO_FREE_GPU" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MC_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

SUMMARY="$RESDIR/curve.tsv";  printf "arm\tselector\tq\ttop_t\tk\tsplit\tloss\tckpt\twandb_url\n" > "$SUMMARY"
RUNDIRS="$RESDIR/rundirs.tsv"; printf "arm\tselector\tq\ttop_t\trun_dir\trc\twall_s\tsolve_s\twandb_url\n" > "$RUNDIRS"

# =============================================================================
# one AM Phase-2 run.  $1 tag  $2 selector  $3 q  $4 top_t  $5 run_name
# =============================================================================
run_arm() {
  local tag="$1" sel="$2" q="$3" topt="$4" runname="$5"
  local log="$RUNLOG/${tag}.log"
  echo "=== RUN ${tag} START $(date -Is) selector=${sel} q=${q} top_t=${topt}"
  local t0; t0=$(date +%s)
  (
    export SLOT_SELECTION="$sel"
    export AM_ROPE_THETA=5000000
    export AM_KEY_REPOSITION=1
    export MAX_QUERIES_PER_HEAD=64
    if [ "$sel" = "constrained_mass" ]; then
      export AM_SAFE_FRACTION="$q"
      export AM_SAFE_METRIC=redundancy
    fi
    USE_IDF=0 \
    GRANULARITY=per_layer \
    TOP_T="$topt" \
    TARGET_MODE=cartridge_plus_doc \
    KEY_MODE=highest_attention \
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
    EVAL_DATA_PATH=data/qasper/eval/qasper_eval_MT.parquet \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    NUM_TOKENS=512 \
    EPOCHS=10 GLOBAL_BATCH_SIZE=32 MAX_STEPS=550 UPDATE_INTERVAL=1 \
    QUERIES_PER_BATCH=all_tokens DISTRIBUTED_BACKEND=nccl \
    ENABLE_OLD_REFERENCE_GUARD=0 OLD_REFERENCE_WEIGHT=1.0 MAX_REF_EXAMPLES_PER_DOC=32 \
    EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
    EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \
    RUN_NAME="$runname" \
    WANDB_DISABLED=0 \
    WANDB_GROUP=B-GATE \
    WANDB_NOTES="$NOTES" \
    "$PY" "$REPO/examples/qasper2/train/continual_am_sparse.py" >"$log" 2>&1
  )
  local rc=$?; local t1; t1=$(date +%s)
  local dir url solve
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  url=$(grep -oE 'https://wandb\.ai/\S+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  solve=$("$PY" -c "
import json
try:
    d=json.load(open('$dir/phase2_summary.json'))
    print(round(float(d.get('am',{}).get('wall_clock_s') or d.get('wall_clock_s') or 0),1))
except Exception: print(0)
" 2>/dev/null)
  echo "RUN ${tag} RC=$rc WALL_S=$((t1-t0)) SOLVE_S=$solve DIR=$dir"
  echo "RUN ${tag} MECH009=$(grep -oE 'MECH-009[^\"]*' "$log" | head -2 | tr '\n' ' ')"
  echo "RUN ${tag} WANDB=$url"
  if [ $rc -ne 0 ]; then echo "RUN ${tag} TRACEBACK_TAIL:"; tail -40 "$log"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$sel" "$q" "$topt" "$dir" "$rc" "$((t1-t0))" "${solve:-0}" "${url:-MISSING}" >> "$RUNDIRS"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

ckpt_for() { ls "$1"/cache-after-doc-$(printf "%03d" $(($2-1)))-*.pt 2>/dev/null | head -1; }

eval_one() {  # arm selector q top_t k split ckpt
  local arm="$1" sel="$2" q="$3" topt="$4" k="$5" split="$6" ckpt="$7"
  local log="$LOGDIR/eval_${arm}_doc${k}_${split}.log"; : > "$log"
  echo "--- [$(date -Is)] EVAL arm=${arm} k=${k} split=${split}"
  (
    CHECKPOINT_PATH="$ckpt" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    RUN_NAME="MECH-CONSTRAINED_${arm}_k${k}_${split}" \
    WANDB_GROUP=B-GATE \
    WANDB_NOTES="$NOTES" \
    WANDB_DISABLED=0 \
    "$PY" "$REPO/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1
  ) &
  local pid=$!
  for _ in $(seq 1 300); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  local loss url ckptok
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  ckptok=$(grep -c -F -- "$ckpt" "$log")
  echo "RESULT arm=${arm} k=${k} ${split}: loss=${loss:-MISSING} ckpt_in_log=${ckptok} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$sel" "$q" "$topt" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK §1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null; sleep 2
}

eval_kcurve() {  # arm selector q top_t run_dir
  local arm="$1" sel="$2" q="$3" topt="$4" dir="$5" K C
  for K in 8 10 12 16; do
    C=$(ckpt_for "$dir" "$K")
    if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=$arm k=$K dir=$dir"; continue; fi
    eval_one "$arm" "$sel" "$q" "$topt" "$K" QA "$C"
    eval_one "$arm" "$sel" "$q" "$topt" "$K" MT "$C"
  done
}

# =============================================================================
# 1. CONTROL + GATE  (SLOT_SELECTION=tfidf must be bit-identical to MECH-005)
# =============================================================================
echo "############ ARM C0: CONTROL / GATE (tfidf, TOP_T=32) ############"
run_arm C0_control tfidf 1.00 32 MECH-CONSTRAINED_q1.00_t32_gate_tfidf ; RC_C0=$?
D_C0=$(cat "$RESDIR/rundir_C0_control.txt")
eval_kcurve C0_control tfidf 1.00 32 "$D_C0"

GATE=$("$PY" - "$SUMMARY" <<'EOF'
import sys
want = {(12,"QA"):1.9560121297836304, (12,"MT"):2.2720184326171875,
        (16,"QA"):2.034916400909424,  (16,"MT"):2.330503463745117}
got, bad = {}, []
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split("\t")
    if f[0] == "C0_control" and f[6] != "MISSING":
        got[(int(f[4]), f[5])] = float(f[6])
for key, w in want.items():
    g = got.get(key)
    if g is None or abs(g - w) > 1e-6:
        bad.append(f"{key}: got {g} want {w}")
print("GATE_FAIL " + " | ".join(bad) if bad else "GATE_PASS")
EOF
)
echo "CONTROL_GATE=$GATE"
if [ "${GATE:0:9}" = "GATE_FAIL" ]; then
  echo "MC_ABORT_GATE_FAILED"; cat "$SUMMARY"; exit 4
fi

# =============================================================================
# 2. CONSTRAINT-STRENGTH SWEEP at TOP_T=32
# =============================================================================
for Q in 0.75 0.50 0.25; do
  TAG="Q${Q#0.}"                      # 0.75 -> Q75, 0.50 -> Q50, 0.25 -> Q25
  echo "############ ARM ${TAG}: constrained_mass q=${Q} redundancy TOP_T=32 ############"
  run_arm "$TAG" constrained_mass "$Q" 32 "MECH-CONSTRAINED_q${Q}_t32"
  D=$(cat "$RESDIR/rundir_${TAG}.txt")
  [ -n "$D" ] && eval_kcurve "$TAG" constrained_mass "$Q" 32 "$D"
done

# =============================================================================
# 3. loosest useful constraint x the least-bad budget MECH-BUDGET found
# =============================================================================
echo "############ ARM Q50T64: constrained_mass q=0.50 redundancy TOP_T=64 ############"
run_arm Q50T64 constrained_mass 0.50 64 MECH-CONSTRAINED_q0.50_t64
D=$(cat "$RESDIR/rundir_Q50T64.txt")
[ -n "$D" ] && eval_kcurve Q50T64 constrained_mass 0.50 64 "$D"

echo "MC_DONE_TS=$(date -Is)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
cat "$RUNDIRS"
cat "$SUMMARY"
