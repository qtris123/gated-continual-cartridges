#!/usr/bin/env bash
# MECH-INFOGATE (W3 BUILD + W4 TEST) — B-GATE / MECH-008.
#
# Information-theoretic slot selection, run as the REVERSE trade DIAG-IMPORTANCE's
# measured Pareto front supports: retention is 0.57 AHEAD of budget while MT is
# 0.248 short, so spend QA slack to buy MT bandwidth by raising `top_t` under a
# redundancy gate, instead of protecting QA at fixed budget.
#
# Arms (all else at the project's current best point):
#   C0  control            SLOT_SELECTION=tfidf              TOP_T=32   <- GATE
#   R32 redundancy         SLOT_SELECTION=redundancy         TOP_T=32
#   R64 redundancy         SLOT_SELECTION=redundancy         TOP_T=64
#   R128 redundancy        SLOT_SELECTION=redundancy         TOP_T=128
#   X<t> mass_x_redundancy SLOT_SELECTION=mass_x_redundancy  TOP_T=<best of R*>
#   F32 fisher             SLOT_SELECTION=fisher             TOP_T=32
#
# Every arm evaluated on BOTH splits at k = 8, 10, 12, 16 (the k=16 endpoint is
# NOT a run's best point: the incumbent bottoms at k=12, the frozen control k=10).
#
# No import pin: this worker IS the source editor and the runs must see its edit.
# PYTHONPATH is pinned to the _explore repo so `import cartridges` cannot land on
# the sibling package (RUNBOOK §6.10 / §9c-bis).
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
PY="$REPO/.venv/bin/python"
RESDIR="$REPO/research_loop/results/MECH-INFOGATE"
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

FISHER_SCORES="$REPO/research_loop/state/diagnostics/slot_fisher_qa_phase1.npz"

echo "MI_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" diff --stat -- cartridges examples > "$RESDIR/worktree_diff_stat.txt" 2>&1
git -C "$REPO" diff -- cartridges examples > "$RESDIR/worktree_diff.patch" 2>&1
echo "WORKTREE_DIFF_STAT:"; cat "$RESDIR/worktree_diff_stat.txt"
echo "IMPORT_PROBE_FROM_TMP=$(cd /tmp && "$PY" -c 'import cartridges,os;print(os.path.dirname(cartridges.__file__))')"
echo "FISHER_SCORES=$FISHER_SCORES exists=$([ -f "$FISHER_SCORES" ] && echo yes || echo no)"

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
  echo "MI_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
[ -z "$CLAIMED" ] && { echo "MI_NO_FREE_GPU" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MI_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

SUMMARY="$RESDIR/curve.tsv";  printf "arm\tselector\ttop_t\tk\tsplit\tloss\tckpt\twandb_url\n" > "$SUMMARY"
RUNDIRS="$RESDIR/rundirs.tsv"; printf "arm\tselector\ttop_t\trun_dir\trc\twall_s\tsolve_s\twandb_url\n" > "$RUNDIRS"

# =============================================================================
# one AM Phase-2 run.  $1 tag  $2 selector  $3 top_t  $4 run_name
# =============================================================================
run_arm() {
  local tag="$1" sel="$2" topt="$3" runname="$4"
  local log="$RUNLOG/${tag}.log"
  echo "=== RUN ${tag} START $(date -Is) selector=${sel} top_t=${topt}"
  local t0; t0=$(date +%s)
  (
    export SLOT_SELECTION="$sel"
    export AM_ROPE_THETA=5000000
    export AM_KEY_REPOSITION=1
    export MAX_QUERIES_PER_HEAD=64
    [ "$sel" = "fisher" ] && export AM_SLOT_FISHER_PATH="$FISHER_SCORES"
    [ "$sel" = "mass_x_redundancy" ] && export AM_MASS_REDUNDANCY_ALPHA=0.5
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
    WANDB_NOTES="information-theoretic slot selection; reverse-trade sweep" \
    "$PY" "$REPO/examples/qasper2/train/continual_am_sparse.py" >"$log" 2>&1
  )
  local rc=$?; local t1; t1=$(date +%s)
  local dir url solve
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  url=$(grep -oE 'https://wandb\.ai/\S+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  solve=$("$PY" -c "
import json,sys
try:
    d=json.load(open('$dir/phase2_summary.json'))
    print(round(float(d.get('am',{}).get('wall_clock_s') or d.get('wall_clock_s') or 0),1))
except Exception: print(0)
" 2>/dev/null)
  echo "RUN ${tag} RC=$rc WALL_S=$((t1-t0)) SOLVE_S=$solve DIR=$dir"
  echo "RUN ${tag} IMPORT=$(grep -oE 'MECH-008:[^\"]*' "$log" | head -1)"
  echo "RUN ${tag} WANDB=$url"
  if [ $rc -ne 0 ]; then echo "RUN ${tag} TRACEBACK_TAIL:"; tail -40 "$log"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$sel" "$topt" "$dir" "$rc" "$((t1-t0))" "${solve:-0}" "${url:-MISSING}" >> "$RUNDIRS"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

ckpt_for() { ls "$1"/cache-after-doc-$(printf "%03d" $(($2-1)))-*.pt 2>/dev/null | head -1; }

eval_one() {  # arm selector top_t k split ckpt
  local arm="$1" sel="$2" topt="$3" k="$4" split="$5" ckpt="$6"
  local log="$LOGDIR/eval_${arm}_doc${k}_${split}.log"; : > "$log"
  echo "--- [$(date -Is)] EVAL arm=${arm} k=${k} split=${split}"
  (
    CHECKPOINT_PATH="$ckpt" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    RUN_NAME="MECH-INFOGATE_${arm}_k${k}_${split}" \
    WANDB_GROUP=B-GATE \
    WANDB_NOTES="information-theoretic slot selection; reverse-trade sweep" \
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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$sel" "$topt" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK §1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null; sleep 2
}

eval_kcurve() {  # arm selector top_t run_dir
  local arm="$1" sel="$2" topt="$3" dir="$4" K C
  for K in 8 10 12 16; do
    C=$(ckpt_for "$dir" "$K")
    if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=$arm k=$K dir=$dir"; continue; fi
    eval_one "$arm" "$sel" "$topt" "$K" QA "$C"
    eval_one "$arm" "$sel" "$topt" "$K" MT "$C"
  done
}

# =============================================================================
# 1. CONTROL + GATE
# =============================================================================
echo "############ ARM C0: CONTROL (tfidf, TOP_T=32) ############"
run_arm C0_control tfidf 32 MECH-INFOGATE_tfidf_t32 ; RC_C0=$?
D_C0=$(cat "$RESDIR/rundir_C0_control.txt")
eval_kcurve C0_control tfidf 32 "$D_C0"

GATE=$("$PY" - "$SUMMARY" <<'EOF'
import sys
want = {(12,"QA"):1.9560121297836304, (12,"MT"):2.2720184326171875,
        (16,"QA"):2.034916400909424,  (16,"MT"):2.330503463745117}
got, bad = {}, []
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split("\t")
    if f[0] == "C0_control" and f[5] != "MISSING":
        got[(int(f[3]), f[4])] = float(f[5])
for key, w in want.items():
    g = got.get(key)
    if g is None or abs(g - w) > 1e-6:
        bad.append(f"{key}: got {g} want {w}")
print("GATE_FAIL " + " | ".join(bad) if bad else "GATE_PASS")
EOF
)
echo "CONTROL_GATE=$GATE"
if [ "${GATE:0:9}" = "GATE_FAIL" ]; then
  echo "MI_ABORT_GATE_FAILED"; cat "$SUMMARY"; exit 4
fi

# =============================================================================
# 2. REDUNDANCY SWEEP — the reverse trade
# =============================================================================
echo "############ ARMS R32 / R64 / R128: redundancy ############"
for T in 32 64 128; do
  run_arm "R${T}" redundancy "$T" "MECH-INFOGATE_redundancy_t${T}"
  D=$(cat "$RESDIR/rundir_R${T}.txt")
  [ -n "$D" ] && eval_kcurve "R${T}" redundancy "$T" "$D"
done

# best redundancy top_t = the one with the lowest MT over k in {8,10,12,16}
BEST_T=$("$PY" - "$SUMMARY" <<'EOF'
import sys, collections
best = collections.defaultdict(lambda: 1e9)
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split("\t")
    if f[1] == "redundancy" and f[4] == "MT" and f[5] != "MISSING":
        best[int(f[2])] = min(best[int(f[2])], float(f[5]))
print(min(best, key=best.get) if best else 32)
EOF
)
echo "BEST_REDUNDANCY_TOP_T=$BEST_T"

# =============================================================================
# 3. mass_x_redundancy at the best redundancy budget
# =============================================================================
echo "############ ARM X${BEST_T}: mass_x_redundancy (alpha=0.5) ############"
run_arm "X${BEST_T}" mass_x_redundancy "$BEST_T" "MECH-INFOGATE_mass_x_redundancy_t${BEST_T}"
D=$(cat "$RESDIR/rundir_X${BEST_T}.txt")
[ -n "$D" ] && eval_kcurve "X${BEST_T}" mass_x_redundancy "$BEST_T" "$D"

# =============================================================================
# 4. fisher at TOP_T=32 — the maximally protective point
# =============================================================================
echo "############ ARM F32: fisher (lowest QA Fisher first) ############"
run_arm F32 fisher 32 MECH-INFOGATE_fisher_t32
D=$(cat "$RESDIR/rundir_F32.txt")
[ -n "$D" ] && eval_kcurve F32 fisher 32 "$D"

echo "MI_DONE_TS=$(date -Is)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
cat "$RUNDIRS"
cat "$SUMMARY"
