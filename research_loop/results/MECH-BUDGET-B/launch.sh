#!/usr/bin/env bash
# MECH-BUDGET-B (W4 TEST) — B-GATE.
#
# REMOVES MECH-BUDGET's OWN CONFOUND.  MECH-BUDGET swept TOP_T in {32,64,128} with
# MAX_QUERIES_PER_HEAD pinned at 64, so top_t=128 solved 64 equations for 128 unknowns
# per head — the data block was rank-deficient and the extra 64 directions of the
# support were pinned to their prior values by the trust region.  Its t128 verdict is
# therefore not a statement about SUPPORT.  Here the large-support arms get enough
# reference queries that the DATA block alone determines the solve.
#
# Arms (env-only; nothing in cartridges/ or examples/ is touched):
#   G_t32q64      TOP_T=32  NQ=64   DW=1e-2   <- GATE.  Must reproduce DIAG-KEYCURVE /
#                                                MECH-BUDGET A_t32:
#                                                k=12 QA 1.9560121297836304 / MT 2.2720184326171875
#                                                k=16 QA 2.034916400909424  / MT 2.330503463745117
#   A_t128q512    TOP_T=128 NQ=512  DW=1e-2   <- brief item 1 (4x over-determined)
#   C_t64q512     TOP_T=64  NQ=512  DW=1e-2   <- brief item 3 (the "more queries alone" control;
#                                                matched n and w with A, so A-C isolates SUPPORT)
#   B_t256q1024   TOP_T=256 NQ=1024 DW=1e-2   <- brief item 2 (ratio held at 4x)
#   D_t128q512dw  TOP_T=128 NQ=512  DW=8e-2   <- sensitivity arm: DELTA_WEIGHT scaled
#                                                w = 1e-2 * n/64 = 0.08 (MECH-QUERIES-B's line),
#                                                so the trust region's RELATIVE pull is held
#                                                at the n=64 value instead of decaying like 1/n.
#
# DELTA_WEIGHT CHOICE (stated, not silent): the four primary arms hold DELTA_WEIGHT at
# the base config's 1e-2.  Reasons: (i) the assigned base config specifies 1e-2 and the
# assigned variable is the query cap; (ii) A vs MECH-BUDGET's C_t128 is then a
# single-variable change (n: 64 -> 512) and A vs C is a single-variable change
# (top_t: 64 -> 128) at MATCHED n and MATCHED w — the trust-region dilution is common to
# both and cancels in that contrast; (iii) MECH-QUERIES-B already measured the w ∝ n line
# at top_t=32 and it did NOT improve MT (+0.046/+0.056), so scaling is not the choice that
# flatters the hypothesis.  D_t128q512dw is the explicit both-ways check.
#
# Every arm evaluated on BOTH splits at k in {8,10,12,16} (ckpt = cache-after-doc-{007,009,011,015}).
#
# IMPORT PIN (RUNBOOK §9c-bis): MECH-INFOGATE is editing cartridges/am/{ranking,continual,
# finetune}.py and the driver.  Driver AND eval AND probe all run from the frozen
# `git archive HEAD` snapshot at /tmp/amsnap_budgetb.  THIS WORKER EDITS NO SOURCE FILE.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_budgetb
PY="$REPO/.venv/bin/python"

RESDIR="$REPO/research_loop/results/MECH-BUDGET-B"
LOGDIR="$RESDIR/evals"
RUNLOG="$RESDIR/runs"
mkdir -p "$LOGDIR" "$RUNLOG"
exec > "$RESDIR/wrapper.log" 2>&1

cd "$REPO" || exit 9

export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0

echo "MBB_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain -- cartridges examples > "$RESDIR/git_status_at_launch.txt" 2>&1
echo "CONCURRENT_EDITS_AT_LAUNCH:"; cat "$RESDIR/git_status_at_launch.txt"

# ---- frozen HEAD snapshot (already built + probed by the worker; rebuild defensively) --
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
  for idx in 1 0; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "MBB_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then echo "MBB_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MBB_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

SUMMARY="$RESDIR/curve.tsv"
: > "$SUMMARY"
printf "arm\ttop_t\tnq\tdw\tk\tsplit\tloss\tckpt\tckpt_sha256\twandb_url\n" >> "$SUMMARY"
RUNDIRS="$RESDIR/rundirs.tsv"
: > "$RUNDIRS"
printf "arm\ttop_t\tnq\tdw\trun_dir\trc\tphase2_e2e_s\twandb_url\n" >> "$RUNDIRS"

# =============================================================================
# one AM Phase-2 run.  $1 tag  $2 top_t  $3 max_queries_per_head  $4 delta_weight
# Base config = the project's current best point (GLOSSARY §4 step 2 / MECH-BUDGET's
# base), SLOT_SELECTION=tfidf.  Only TOP_T / MAX_QUERIES_PER_HEAD (/ DELTA_WEIGHT in
# the D arm) vary.
# =============================================================================
run_arm() {
  local tag="$1" topt="$2" nq="$3" dw="$4"
  local log="$RUNLOG/${tag}.log"
  local runname="MECH-BUDGET-B_t${topt}_q${nq}"
  [ "$dw" = "1e-2" ] || runname="${runname}_dw${dw}"
  echo "=== RUN ${tag} START $(date -Is) TOP_T=${topt} NQ=${nq} DW=${dw} run_name=${runname}"
  local t0=$(date +%s)
  (
    export CARTRIDGES_DIR="$SNAP"
    export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
    export AM_ROPE_THETA=5000000
    export AM_KEY_REPOSITION=1
    export MAX_QUERIES_PER_HEAD="$nq"
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
    DELTA_WEIGHT="$dw" \
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
    WANDB_NOTES="large support with a determined solve — removing MECH-BUDGET's underdetermination confound" \
    "$PY" "$SNAP/examples/qasper2/train/continual_am_sparse.py" >"$log" 2>&1
  )
  local rc=$?
  local t1=$(date +%s)
  local dir url
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  url=$(grep -oE 'https://wandb\.ai/\S+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  echo "RUN ${tag} RC=$rc PHASE2_E2E_S=$((t1-t0)) DIR=$dir"
  echo "RUN ${tag} IMPORT=$(grep -oE '/(tmp/amsnap_budgetb|localhome/local-triv/gated-continual-cartridges[a-z_]*)/cartridges' "$log" | head -1)"
  echo "RUN ${tag} WANDB=$url"
  echo "RUN ${tag} SUMMARY=$(tr -d '\n ' < "$dir/phase2_summary.json" 2>/dev/null)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$topt" "$nq" "$dw" "$dir" "$rc" "$((t1-t0))" "${url:-MISSING}" >> "$RUNDIRS"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  echo "$((t1-t0))" > "$RESDIR/e2e_${tag}.txt"
  return $rc
}

ckpt_for() {  # run_dir k -> cache-after-doc-<k-1>-*.pt
  local dir="$1" k="$2"
  ls "$dir"/cache-after-doc-$(printf "%03d" $((k-1)))-*.pt 2>/dev/null | head -1
}

eval_one() {  # arm top_t nq dw k split ckpt
  local arm="$1" topt="$2" nq="$3" dw="$4" k="$5" split="$6" ckpt="$7"
  local log="$LOGDIR/eval_${arm}_k${k}_${split}.log"
  : > "$log"
  echo "--- [$(date -Is)] EVAL arm=${arm} top_t=${topt} nq=${nq} k=${k} split=${split} ckpt=${ckpt}"
  (
    export CARTRIDGES_DIR="$SNAP"
    export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
    CHECKPOINT_PATH="$ckpt" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    RUN_NAME="MECH-BUDGET-B_t${topt}_q${nq}_k${k}_${split}" \
    WANDB_GROUP=B-GATE \
    WANDB_NOTES="large support with a determined solve — removing MECH-BUDGET's underdetermination confound" \
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
  echo "RESULT arm=${arm} top_t=${topt} nq=${nq} k=${k} ${split}: loss=${loss:-MISSING} ckpt_in_log=${ckptok} sha=${sha} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$arm" "$topt" "$nq" "$dw" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${sha:-MISSING}" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK §1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

eval_kcurve() {  # arm top_t nq dw run_dir  -> both splits at k in {8,10,12,16}
  local arm="$1" topt="$2" nq="$3" dw="$4" dir="$5"
  local K C
  for K in 8 10 12 16; do
    C=$(ckpt_for "$dir" "$K")
    if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=$arm k=$K dir=$dir"; continue; fi
    eval_one "$arm" "$topt" "$nq" "$dw" "$K" QA "$C"
    eval_one "$arm" "$topt" "$nq" "$dw" "$K" MT "$C"
  done
}

do_arm() {  # tag top_t nq dw
  local tag="$1" topt="$2" nq="$3" dw="$4"
  echo "############ ARM ${tag}: TOP_T=${topt} NQ=${nq} DW=${dw} ############"
  run_arm "$tag" "$topt" "$nq" "$dw"; local rc=$?
  local D
  D=$(cat "$RESDIR/rundir_${tag}.txt" 2>/dev/null)
  if [ -n "$D" ] && [ -d "$D" ]; then
    eval_kcurve "$tag" "$topt" "$nq" "$dw" "$D"
  else
    echo "MBB_NO_RUNDIR_${tag} rc=$rc"; tail -40 "$RUNLOG/${tag}.log"
  fi
  return $rc
}

# =============================================================================
# ARM G — the GATE (unchanged base config; must reproduce to 1e-6)
# =============================================================================
do_arm G_t32q64 32 64 1e-2 ; RC_G=$?
D_G=$(cat "$RESDIR/rundir_G_t32q64.txt" 2>/dev/null)
if [ -z "$D_G" ] || [ ! -d "$D_G" ]; then echo "MBB_ABORT_NO_RUNDIR_G rc=$RC_G"; exit 5; fi

GATE=$("$PY" - "$SUMMARY" <<'EOF'
import sys
want = {(12,"QA"):1.9560121297836304, (12,"MT"):2.2720184326171875,
        (16,"QA"):2.034916400909424,  (16,"MT"):2.330503463745117}
got, bad = {}, []
for line in open(sys.argv[1]).read().splitlines()[1:]:
    p = line.split("\t")
    arm, k, sp, loss = p[0], p[4], p[5], p[6]
    if arm == "G_t32q64" and loss != "MISSING":
        got[(int(k), sp)] = float(loss)
for key, w in want.items():
    g = got.get(key)
    if g is None or abs(g - w) > 1e-6:
        bad.append(f"{key}: got {g} want {w}")
print("GATE_FAIL " + " | ".join(bad) if bad else "GATE_PASS")
EOF
)
echo "GATE=$GATE"
if [ "${GATE:0:9}" = "GATE_FAIL" ]; then
  echo "MBB_ABORT_GATE_FAILED"
  cat "$SUMMARY"
  echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
  exit 4
fi

# =============================================================================
# The determined large-support arms, in priority order.
# =============================================================================
do_arm A_t128q512  128 512  1e-2 ; RC_A=$?
do_arm C_t64q512    64 512  1e-2 ; RC_C=$?
do_arm B_t256q1024 256 1024 1e-2 ; RC_B=$?
do_arm D_t128q512dw 128 512 8e-2 ; RC_D=$?

# =============================================================================
# EVAL-TIME ROUTING MASS (mass_on_S, total cartridge mass, MT/QA ratio) per arm
# standalone probe, byte-identical copy of results/MECH-BUDGET/measure_route_mass.py
# =============================================================================
echo "############ EVAL-TIME mass_on_S PROBE ############"
SPECS=""
for tag in G_t32q64 A_t128q512 C_t64q512 B_t256q1024 D_t128q512dw; do
  d=$(cat "$RESDIR/rundir_${tag}.txt" 2>/dev/null)
  [ -n "$d" ] && [ -d "$d" ] || continue
  c=$(ls "$d"/cache-after-doc-015-*.pt 2>/dev/null | head -1)
  [ -n "$c" ] || continue
  SPECS="${SPECS:+$SPECS,}${tag}:${c}:${d}"
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
  for i in $(seq 1 1200); do
    grep -q "\[done\] wrote" "$RESDIR/route_mass.log" && break
    kill -0 "$RMPID" 2>/dev/null || break
    sleep 3
  done
  echo "ROUTE_MASS_TAIL:"; tail -20 "$RESDIR/route_mass.log"
  kill -TERM "$RMPID" 2>/dev/null; sleep 3; kill -KILL "$RMPID" 2>/dev/null
fi

echo "MBB_RC G=$RC_G A=${RC_A:-NA} C=${RC_C:-NA} B=${RC_B:-NA} D=${RC_D:-NA}"
MANIFEST_AFTER=$(manifest)
echo "SNAPSHOT_MANIFEST_AFTER=$MANIFEST_AFTER"
echo "SNAPSHOT_MANIFEST_STABLE=$([ "$MANIFEST_BEFORE" = "$MANIFEST_AFTER" ] && echo yes || echo no)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain -- cartridges examples > "$RESDIR/git_status_after.txt" 2>&1
echo "MBB_DONE_TS=$(date -Is)"
cat "$SUMMARY"
