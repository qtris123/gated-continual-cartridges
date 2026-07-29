#!/usr/bin/env bash
# MECH-BUDGET (W4 TEST) — B-GATE.
#
# THE MISSING CONTROL FOR MECH-INFOGATE: the INCUMBENT selector (SLOT_SELECTION=tfidf)
# run at LARGER top_t.  MECH-INFOGATE's only control is tfidf@32, which confounds
# "does a bigger write budget help?" with "does smarter selection help?".
#
# Arms (env-only; ONE variable changes: TOP_T):
#   A_t32  TOP_T=32   <- GATE.  Must reproduce DIAG-KEYCURVE:
#                       k=16 QA 2.034916400909424 / MT 2.330503463745117
#                       k=12 QA 1.9560121297836304 / MT 2.2720184326171875
#   B_t64  TOP_T=64
#   C_t128 TOP_T=128
#
# Every arm evaluated on BOTH splits at k in {8,10,12,16}
# (ckpt = cache-after-doc-{007,009,011,015}-*.pt).
#
# IMPORT PIN (RUNBOOK §9c-bis): MECH-INFOGATE is editing cartridges/am/{ranking,continual,
# finetune}.py and the driver RIGHT NOW.  Everything here — driver AND eval AND probes —
# runs from the frozen `git archive HEAD` snapshot at /tmp/amsnap_budget.
# THIS WORKER EDITS NO SOURCE FILE.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_budget
PY="$REPO/.venv/bin/python"

RESDIR="$REPO/research_loop/results/MECH-BUDGET"
LOGDIR="$RESDIR/evals"
RUNLOG="$RESDIR/runs"
mkdir -p "$LOGDIR" "$RUNLOG"
exec > "$RESDIR/wrapper.log" 2>&1

cd "$REPO" || exit 9

export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0

echo "MB_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain -- cartridges examples > "$RESDIR/git_status_at_launch.txt" 2>&1
echo "CONCURRENT_EDITS_AT_LAUNCH:"; cat "$RESDIR/git_status_at_launch.txt"

# ---- frozen HEAD snapshot (already built by the worker; rebuild defensively) --
rm -rf "$SNAP"; mkdir -p "$SNAP"
git -C "$REPO" archive HEAD cartridges examples | tar -x -C "$SNAP"
ln -sfn "$REPO/.venv" "$SNAP/.venv"
ln -sfn "$REPO/data"  "$SNAP/data"
manifest() { (cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort \
  | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1); }
echo "SNAPSHOT_PATH=$SNAP"
MANIFEST_BEFORE=$(manifest)
echo "SNAPSHOT_MANIFEST_AT_LAUNCH=$MANIFEST_BEFORE"
# §9c-bis: probe from /tmp, NOT the repo root (cwd is sys.path[0] -> false negative)
echo "IMPORT_PROBE=$(cd /tmp && PYTHONPATH="$SNAP" CARTRIDGES_DIR="$SNAP" "$PY" -c \
  'import cartridges,os;print(os.path.dirname(cartridges.__file__))' 2>&1 | tail -1)"

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
  echo "MB_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then echo "MB_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MB_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

SUMMARY="$RESDIR/curve.tsv"
: > "$SUMMARY"
printf "arm\ttop_t\tk\tsplit\tloss\tckpt\tckpt_sha256\twandb_url\n" >> "$SUMMARY"
RUNDIRS="$RESDIR/rundirs.tsv"
: > "$RUNDIRS"
printf "arm\ttop_t\trun_dir\trc\tphase2_e2e_s\twandb_url\n" >> "$RUNDIRS"

# =============================================================================
# one AM Phase-2 run.  $1 tag  $2 top_t
# Base config = the project's current best point (GLOSSARY §4 step 2), with
# SLOT_SELECTION=tfidf (the INCUMBENT).  Only TOP_T varies across arms.
# =============================================================================
run_arm() {
  local tag="$1" topt="$2"
  local log="$RUNLOG/${tag}.log"
  local runname="MECH-BUDGET_tfidf_t${topt}"
  echo "=== RUN ${tag} START $(date -Is) TOP_T=${topt} run_name=${runname}"
  local t0=$(date +%s)
  (
    export CARTRIDGES_DIR="$SNAP"
    export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
    export AM_ROPE_THETA=5000000
    export AM_KEY_REPOSITION=1
    export MAX_QUERIES_PER_HEAD=64
    export SLOT_SELECTION=tfidf
    USE_IDF=0 \
    GRANULARITY=per_layer \
    TOP_T="$topt" \
    MIN_TOP_T_PER_LAYER=1 \
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
    WANDB_NOTES="incumbent selector at larger top_t — the missing control for MECH-INFOGATE" \
    "$PY" "$SNAP/examples/qasper2/train/continual_am_sparse.py" >"$log" 2>&1
  )
  local rc=$?
  local t1=$(date +%s)
  local dir url
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  url=$(grep -oE 'https://wandb\.ai/\S+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  echo "RUN ${tag} RC=$rc PHASE2_E2E_S=$((t1-t0)) DIR=$dir"
  echo "RUN ${tag} IMPORT=$(grep -oE '/(tmp/amsnap_budget|localhome/local-triv/gated-continual-cartridges[a-z_]*)/cartridges' "$log" | head -1)"
  echo "RUN ${tag} WANDB=$url"
  echo "RUN ${tag} SUMMARY=$(tr -d '\n ' < "$dir/phase2_summary.json" 2>/dev/null)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$topt" "$dir" "$rc" "$((t1-t0))" "${url:-MISSING}" >> "$RUNDIRS"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  echo "$((t1-t0))" > "$RESDIR/e2e_${tag}.txt"
  return $rc
}

ckpt_for() {  # run_dir k -> cache-after-doc-<k-1>-*.pt
  local dir="$1" k="$2"
  ls "$dir"/cache-after-doc-$(printf "%03d" $((k-1)))-*.pt 2>/dev/null | head -1
}

eval_one() {  # arm top_t k split ckpt
  local arm="$1" topt="$2" k="$3" split="$4" ckpt="$5"
  local log="$LOGDIR/eval_${arm}_k${k}_${split}.log"
  : > "$log"
  echo "--- [$(date -Is)] EVAL arm=${arm} top_t=${topt} k=${k} split=${split} ckpt=${ckpt}"
  (
    export CARTRIDGES_DIR="$SNAP"
    export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
    CHECKPOINT_PATH="$ckpt" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    RUN_NAME="MECH-BUDGET_tfidf_t${topt}_k${k}_${split}" \
    WANDB_GROUP=B-GATE \
    WANDB_NOTES="incumbent selector at larger top_t — the missing control for MECH-INFOGATE" \
    WANDB_DISABLED=0 \
    "$PY" "$SNAP/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1
  ) &
  local pid=$!
  local i
  for i in $(seq 1 300); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  local loss url ckptok sha
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  ckptok=$(grep -c -F -- "$ckpt" "$log")
  sha=$(sha256sum "$ckpt" 2>/dev/null | cut -c1-16)
  echo "RESULT arm=${arm} top_t=${topt} k=${k} ${split}: loss=${loss:-MISSING} ckpt_in_log=${ckptok} sha=${sha} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$arm" "$topt" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${sha:-MISSING}" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK §1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

eval_kcurve() {  # arm top_t run_dir  -> both splits at k in {8,10,12,16}
  local arm="$1" topt="$2" dir="$3"
  local K C
  for K in 8 10 12 16; do
    C=$(ckpt_for "$dir" "$K")
    if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=$arm k=$K dir=$dir"; continue; fi
    eval_one "$arm" "$topt" "$K" QA "$C"
    eval_one "$arm" "$topt" "$K" MT "$C"
  done
}

# =============================================================================
# ARM A — TOP_T=32  (THE GATE)
# =============================================================================
echo "############ ARM A: TOP_T=32 (GATE) ############"
run_arm A_t32 32 ; RC_A=$?
D_A=$(cat "$RESDIR/rundir_A_t32.txt" 2>/dev/null)
if [ -z "$D_A" ] || [ ! -d "$D_A" ]; then echo "MB_ABORT_NO_RUNDIR_A rc=$RC_A"; tail -40 "$RUNLOG/A_t32.log"; exit 5; fi
eval_kcurve A_t32 32 "$D_A"

GATE=$("$PY" - "$SUMMARY" <<'EOF'
import sys
want = {(12,"QA"):1.9560121297836304, (12,"MT"):2.2720184326171875,
        (16,"QA"):2.034916400909424,  (16,"MT"):2.330503463745117}
got, bad = {}, []
for line in open(sys.argv[1]).read().splitlines()[1:]:
    p = line.split("\t")
    arm, topt, k, sp, loss = p[0], p[1], p[2], p[3], p[4]
    if arm == "A_t32" and loss != "MISSING":
        got[(int(k), sp)] = float(loss)
for key, w in want.items():
    g = got.get(key)
    if g is None or abs(g - w) > 1e-6:
        bad.append(f"{key}: got {g} want {w}")
print("GATE_FAIL " + " | ".join(bad) if bad else "GATE_PASS")
EOF
)
echo "TOP_T32_GATE=$GATE"
if [ "${GATE:0:9}" = "GATE_FAIL" ]; then
  echo "MB_ABORT_GATE_FAILED"
  cat "$SUMMARY"
  echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
  exit 4
fi

# =============================================================================
# ARM B — TOP_T=64 ; ARM C — TOP_T=128
# =============================================================================
echo "############ ARM B: TOP_T=64 ############"
run_arm B_t64 64 ; RC_B=$?
D_B=$(cat "$RESDIR/rundir_B_t64.txt" 2>/dev/null)
if [ -n "$D_B" ] && [ -d "$D_B" ]; then eval_kcurve B_t64 64 "$D_B"; else echo "MB_NO_RUNDIR_B rc=$RC_B"; tail -40 "$RUNLOG/B_t64.log"; fi

echo "############ ARM C: TOP_T=128 ############"
run_arm C_t128 128 ; RC_C=$?
D_C=$(cat "$RESDIR/rundir_C_t128.txt" 2>/dev/null)
if [ -n "$D_C" ] && [ -d "$D_C" ]; then eval_kcurve C_t128 128 "$D_C"; else echo "MB_NO_RUNDIR_C rc=$RC_C"; tail -40 "$RUNLOG/C_t128.log"; fi

# =============================================================================
# EVAL-TIME ROUTING MASS (mass_on_S, total cartridge mass, MT/QA ratio) per arm
# standalone probe, copied from results/MECH-KEYS/measure_route_mass.py (unmodified)
# =============================================================================
echo "############ EVAL-TIME mass_on_S PROBE ############"
SPECS=""
for spec in "t32:$D_A" "t64:$D_B" "t128:$D_C"; do
  lbl="${spec%%:*}"; d="${spec#*:}"
  [ -n "$d" ] && [ -d "$d" ] || continue
  c=$(ls "$d"/cache-after-doc-015-*.pt 2>/dev/null | head -1)
  [ -n "$c" ] || continue
  SPECS="${SPECS:+$SPECS,}${lbl}:${c}:${d}"
done
echo "ROUTE_MASS_SPECS=$SPECS"
if [ -n "$SPECS" ]; then
  (
    export CARTRIDGES_DIR="$SNAP"
    export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
    CACHE_SPECS="$SPECS" \
    EVAL_SPECS="QA:$REPO/data/qasper/eval/qasper_eval_QA.parquet,MT:$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
    OUT_JSON="$RESDIR/route_mass.json" \
    "$PY" "$RESDIR/measure_route_mass.py" > "$RESDIR/route_mass.log" 2>&1
  ) &
  RMPID=$!
  for i in $(seq 1 900); do
    grep -q "\[done\] wrote" "$RESDIR/route_mass.log" && break
    kill -0 "$RMPID" 2>/dev/null || break
    sleep 3
  done
  echo "ROUTE_MASS_TAIL:"; tail -20 "$RESDIR/route_mass.log"
  kill -TERM "$RMPID" 2>/dev/null; sleep 3; kill -KILL "$RMPID" 2>/dev/null
fi

echo "MB_RC A=$RC_A B=${RC_B:-NA} C=${RC_C:-NA}"
MANIFEST_AFTER=$(manifest)
echo "SNAPSHOT_MANIFEST_AFTER=$MANIFEST_AFTER"
echo "SNAPSHOT_MANIFEST_STABLE=$([ "$MANIFEST_BEFORE" = "$MANIFEST_AFTER" ] && echo yes || echo no)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain -- cartridges examples > "$RESDIR/git_status_after.txt" 2>&1
echo "MB_DONE_TS=$(date -Is)"
cat "$SUMMARY"
