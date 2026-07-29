#!/usr/bin/env bash
# VERIFY-GATE (W5 VERIFY) — B-GATE. Try to KILL MECH-INFOGATE's negative result.
#
# MECH-INFOGATE closed B-GATE on ONE SEED PER ARM: redundancy@t32 lost acquisition
# by dMT +0.167 (3.4x DIAG-NOISE's paired resolution +-0.049). MECH-SEED showed the
# seed knob (AM_SEED_OFFSET, MECH-007) replaces ~30 of each document's 32 reference
# conversations, so it is a genuine perturbation. If the gating negative does not
# survive it, the human's hypothesis is back open.
#
# ARMS (everything else at the project's best point: KEY_MODE=highest_attention,
#   AM_KEY_REPOSITION=1, AM_ROPE_THETA=5000000, ENABLE_BETA=0, GRANULARITY=per_layer,
#   USE_IDF=0, TOP_T=32, RIDGE_LAMBDA=1e-4, RIDGE_SCALE=spectral, DELTA_WEIGHT=1e-2,
#   MAX_QUERIES_PER_HEAD=64):
#   R_off0    SLOT_SELECTION=redundancy AM_SEED_OFFSET=0     <- THE GATE
#   R_off1000 SLOT_SELECTION=redundancy AM_SEED_OFFSET=1000
#   R_off2000 SLOT_SELECTION=redundancy AM_SEED_OFFSET=2000
#   T_off1000 SLOT_SELECTION=tfidf      AM_SEED_OFFSET=1000  <- REUSED CHECKPOINTS (MECH-SEED B1)
#   T_off2000 SLOT_SELECTION=tfidf      AM_SEED_OFFSET=2000  <- REUSED CHECKPOINTS (MECH-SEED B2)
#   T_off0    reused published k-curve (MECH-INFOGATE C0_control) -- no run, no eval
#
# Both splits at k in {8,10,12,16} for every arm evaluated here.
#
# IMPORT PIN (RUNBOOK 9c-bis): MECH-CONSTRAINED is editing cartridges/am/ranking.py
# RIGHT NOW. Everything below imports from a `git archive HEAD` snapshot in /tmp and
# probes the resolved package FROM /tmp (never the repo root -- that probe is a false
# negative). Manifest hashed before AND after.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_verifygate
PY="$REPO/.venv/bin/python"

RESDIR="$REPO/research_loop/results/VERIFY-GATE"
LOGDIR="$RESDIR/evals"; RUNLOG="$RESDIR/runs"
mkdir -p "$LOGDIR" "$RUNLOG"
exec > "$RESDIR/wrapper.log" 2>&1

cd "$REPO" || exit 9
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0

echo "VG_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain -- cartridges examples > "$RESDIR/git_status_at_launch.txt" 2>&1
echo "WORKTREE_DIRTY_AT_LAUNCH:"; cat "$RESDIR/git_status_at_launch.txt"

# ---- frozen HEAD snapshot ----------------------------------------------------
rm -rf "$SNAP"; mkdir -p "$SNAP"
git -C "$REPO" archive HEAD cartridges examples | tar -x -C "$SNAP"
ln -sfn "$REPO/.venv" "$SNAP/.venv"
ln -sfn "$REPO/data"  "$SNAP/data"
manifest() { (cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort \
  | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1); }
MAN_BEFORE=$(manifest)
echo "SNAPSHOT_PATH=$SNAP"
echo "SNAPSHOT_MANIFEST_BEFORE=$MAN_BEFORE"
echo "$MAN_BEFORE" > "$RESDIR/snapshot_manifest_before.txt"
echo "SNAPSHOT_HAS_MECH008=$(grep -c 'SLOT_PRIOR_SELECTIONS' "$SNAP/cartridges/am/ranking.py")"
echo "SNAPSHOT_HAS_MECH007=$(grep -c 'seed_offset' "$SNAP/cartridges/am/continual.py")"
echo "SNAPSHOT_HAS_MECH009_constrained_mass=$(grep -c 'constrained_mass' "$SNAP/cartridges/am/ranking.py")"
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
echo "IMPORT_PROBE_FROM_TMP=$(cd /tmp && CARTRIDGES_DIR="$SNAP" "$PY" -c 'import cartridges,os;print(os.path.dirname(cartridges.__file__))')"

# ---- claim GPU 1 via flock (gpu0 is held by MECH-CONSTRAINED; never steal) ----
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 480); do
  exec {fd}>"$GPU_LOCK_DIR/gpu1.lock"
  if flock -n "$fd"; then CLAIMED=1; break; else exec {fd}>&-; fi
  echo "VG_WAITING_FOR_GPU1 attempt=$attempt ts=$(date -Is)"
  sleep 20
done
[ -z "$CLAIMED" ] && { echo "VG_NO_FREE_GPU" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "VG_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

SUMMARY="$RESDIR/curve.tsv"; printf "arm\tselector\toffset\tk\tsplit\tloss\tckpt\twandb_url\n" > "$SUMMARY"
RUNDIRS="$RESDIR/rundirs.tsv"; printf "arm\tselector\toffset\trun_dir\trc\twall_s\tsolve_s\twandb_url\n" > "$RUNDIRS"

# =============================================================================
# one AM Phase-2 run.  $1 tag  $2 selector  $3 seed_offset  $4 run_name
# =============================================================================
run_arm() {
  local tag="$1" sel="$2" soff="$3" runname="$4"
  local log="$RUNLOG/${tag}.log"
  echo "=== RUN ${tag} START $(date -Is) selector=${sel} seed_offset=${soff}"
  local t0; t0=$(date +%s)
  (
    export CARTRIDGES_DIR="$SNAP"
    export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
    export SLOT_SELECTION="$sel"
    export AM_SEED_OFFSET="$soff"
    export AM_ROPE_THETA=5000000
    export AM_KEY_REPOSITION=1
    export MAX_QUERIES_PER_HEAD=64
    USE_IDF=0 \
    GRANULARITY=per_layer \
    TOP_T=32 \
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
    WANDB_NOTES="seed-varying the gating negative" \
    "$PY" "$SNAP/examples/qasper2/train/continual_am_sparse.py" >"$log" 2>&1
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
  echo "RUN ${tag} IMPORT=$(grep -oE '/(tmp/amsnap_verifygate|localhome/local-triv/gated-continual-cartridges[a-z_-]*)/cartridges' "$log" | head -1)"
  echo "RUN ${tag} SEEDLOG=$(grep -oE 'MECH-007:[^\"]*' "$log" | head -2 | tr '\n' ' ')"
  echo "RUN ${tag} MECH008LOG=$(grep -oE 'MECH-008:[^\"]*' "$log" | head -2 | tr '\n' ' ')"
  echo "RUN ${tag} WANDB=$url"
  if [ $rc -ne 0 ]; then echo "RUN ${tag} TRACEBACK_TAIL:"; tail -50 "$log"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$sel" "$soff" "$dir" "$rc" "$((t1-t0))" "${solve:-0}" "${url:-MISSING}" >> "$RUNDIRS"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

ckpt_for() { ls "$1"/cache-after-doc-$(printf "%03d" $(($2-1)))-*.pt 2>/dev/null | head -1; }

eval_one() {  # arm selector offset k split ckpt
  local arm="$1" sel="$2" off="$3" k="$4" split="$5" ckpt="$6"
  local log="$LOGDIR/eval_${arm}_k${k}_${split}.log"; : > "$log"
  echo "--- [$(date -Is)] EVAL arm=${arm} off=${off} k=${k} split=${split}"
  (
    export CARTRIDGES_DIR="$SNAP"
    export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
    CHECKPOINT_PATH="$ckpt" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    RUN_NAME="VERIFY-GATE_${sel}_off${off}_k${k}_${split}" \
    WANDB_GROUP=B-GATE \
    WANDB_NOTES="seed-varying the gating negative" \
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
  local loss url ckptok
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  ckptok=$(grep -c -F -- "$ckpt" "$log")
  echo "RESULT arm=${arm} off=${off} k=${k} ${split}: loss=${loss:-MISSING} ckpt_in_log=${ckptok} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$sel" "$off" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK 1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null; sleep 2
}

eval_kcurve() {  # arm selector offset run_dir
  local arm="$1" sel="$2" off="$3" dir="$4" K C
  for K in 8 10 12 16; do
    C=$(ckpt_for "$dir" "$K")
    if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=$arm k=$K dir=$dir"; continue; fi
    eval_one "$arm" "$sel" "$off" "$K" QA "$C"
    eval_one "$arm" "$sel" "$off" "$K" MT "$C"
  done
}

hashes_of() {
  local dir="$1" f
  (cd "$dir" && for f in $(ls cache-after-doc-*.pt 2>/dev/null | LC_ALL=C sort); do
      [ -r "$f" ] && sha256sum "$f" | awk '{print $1"  "$2}'
    done)
}

# =============================================================================
# 1. THE GATE — redundancy at AM_SEED_OFFSET=0 must reproduce MECH-INFOGATE R32
# =============================================================================
echo "############ ARM R_off0: redundancy, seed offset 0 (GATE) ############"
run_arm R_off0 redundancy 0 VERIFY-GATE_redundancy_off0
D_R0=$(cat "$RESDIR/rundir_R_off0.txt")
if [ -z "$D_R0" ]; then echo "VG_ABORT_NO_RUNDIR_R_off0"; exit 5; fi
eval_kcurve R_off0 redundancy 0 "$D_R0"

# bytes-level gate: is this cartridge sha256-identical to MECH-INFOGATE's R32?
MI_R32=/localhome/local-triv/gated-continual-cartridges_explore/outputs/2026-07-29-07-52-51-continual_am_sparse/ef58d93f-2b0f-47b8-98a9-2fcde6f59484
hashes_of "$D_R0"   > "$RESDIR/hashes_R_off0.txt"
hashes_of "$MI_R32" > "$RESDIR/hashes_MECH-INFOGATE_R32.txt"
if diff -q "$RESDIR/hashes_R_off0.txt" "$RESDIR/hashes_MECH-INFOGATE_R32.txt" >/dev/null; then
  echo "BYTES_GATE_R_off0_VS_MECH-INFOGATE_R32=IDENTICAL n=$(wc -l < "$RESDIR/hashes_R_off0.txt")"
else
  echo "BYTES_GATE_R_off0_VS_MECH-INFOGATE_R32=DIFFER"
  diff "$RESDIR/hashes_R_off0.txt" "$RESDIR/hashes_MECH-INFOGATE_R32.txt" | head -20
fi

GATE=$("$PY" - "$SUMMARY" <<'EOF'
import sys
want = {(8,"QA"):1.930572748184204,  (8,"MT"):2.5448672771453857,
        (10,"QA"):1.9247082471847534,(10,"MT"):2.522254705429077,
        (12,"QA"):1.9499645233154297,(12,"MT"):2.478079080581665,
        (16,"QA"):1.9893748760223389,(16,"MT"):2.4386465549468994}
got, bad = {}, []
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split("\t")
    if f[0] == "R_off0" and f[5] != "MISSING":
        got[(int(f[3]), f[4])] = float(f[5])
for key, w in want.items():
    g = got.get(key)
    if g is None or abs(g - w) > 1e-6:
        bad.append(f"{key}: got {g} want {w}")
print("GATE_FAIL " + " | ".join(bad) if bad else "GATE_PASS")
EOF
)
echo "REDUNDANCY_OFF0_GATE=$GATE"
if [ "${GATE:0:9}" = "GATE_FAIL" ]; then
  echo "VG_ABORT_GATE_FAILED"; cat "$SUMMARY"; echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"; exit 4
fi

# =============================================================================
# 2. redundancy at the two non-zero offsets
# =============================================================================
for OFF in 1000 2000; do
  echo "############ ARM R_off${OFF}: redundancy, seed offset ${OFF} ############"
  run_arm "R_off${OFF}" redundancy "$OFF" "VERIFY-GATE_redundancy_off${OFF}"
  D=$(cat "$RESDIR/rundir_R_off${OFF}.txt")
  if [ -n "$D" ]; then eval_kcurve "R_off${OFF}" redundancy "$OFF" "$D"; else echo "MISSING_RUNDIR R_off${OFF}"; fi
done

# =============================================================================
# 3. tfidf control at the two non-zero offsets — CHECKPOINTS ALREADY EXIST
#    (MECH-SEED B1/B2). Nothing is retrained; the full k-curve is re-evaluated in
#    THIS session so every number in the comparison comes from one harness path.
# =============================================================================
T1000=/localhome/local-triv/gated-continual-cartridges_explore/outputs/2026-07-29-04-40-40-continual_am_sparse/92b047bb-8822-4f50-8720-5dc2fcb26a4f
T2000=/localhome/local-triv/gated-continual-cartridges_explore/outputs/2026-07-29-05-06-27-continual_am_sparse/5d1785dc-3bc8-40d4-b177-12b7a20f1743
echo "$T1000" > "$RESDIR/rundir_T_off1000.txt"
echo "$T2000" > "$RESDIR/rundir_T_off2000.txt"
printf "T_off1000\ttfidf\t1000\t%s\t0\t437\t296.8\tREUSED_MECH-SEED_B1\n" "$T1000" >> "$RUNDIRS"
printf "T_off2000\ttfidf\t2000\t%s\t0\t669\t310.5\tREUSED_MECH-SEED_B2\n" "$T2000" >> "$RUNDIRS"
echo "############ ARM T_off1000: tfidf control, seed offset 1000 (reused ckpts) ############"
eval_kcurve T_off1000 tfidf 1000 "$T1000"
echo "############ ARM T_off2000: tfidf control, seed offset 2000 (reused ckpts) ############"
eval_kcurve T_off2000 tfidf 2000 "$T2000"

# reproduction check against MECH-SEED / VERIFY-OPTIMA published values
"$PY" - "$SUMMARY" <<'EOF'
import sys
pub = {
 ("T_off1000",10,"MT"):2.273444652557373,  ("T_off1000",10,"QA"):1.911603569984436,
 ("T_off1000",12,"MT"):2.269538164138794,  ("T_off1000",12,"QA"):1.9397201538085938,
 ("T_off1000",16,"MT"):2.306238889694214,  ("T_off1000",16,"QA"):1.996092677116394,
 ("T_off2000",10,"MT"):2.303954601287842,  ("T_off2000",10,"QA"):1.9478639364242554,
 ("T_off2000",12,"MT"):2.262629747390747,  ("T_off2000",12,"QA"):1.9446364641189575,
 ("T_off2000",16,"MT"):2.290109395980835,  ("T_off2000",16,"QA"):2.0024614334106445,
}
got = {}
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split("\t")
    if f[5] != "MISSING":
        got[(f[0], int(f[3]), f[4])] = float(f[5])
ok = True
for key, w in sorted(pub.items()):
    g = got.get(key)
    tag = "EXACT" if (g is not None and abs(g-w) <= 1e-9) else ("CLOSE" if (g is not None and abs(g-w) <= 1e-6) else "MISMATCH")
    if tag == "MISMATCH": ok = False
    print(f"TFIDF_REPRO_GATE {key}: got {g} want {w} -> {tag}")
print("TFIDF_REPRO_GATE_OVERALL=" + ("PASS" if ok else "FAIL"))
EOF

echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
manifest > "$RESDIR/snapshot_manifest_after.txt"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" status --porcelain -- cartridges examples > "$RESDIR/git_status_after.txt" 2>&1
echo "VG_DONE_TS=$(date -Is)"
cat "$RUNDIRS"
cat "$SUMMARY"
echo "VG_ALL_DONE"
