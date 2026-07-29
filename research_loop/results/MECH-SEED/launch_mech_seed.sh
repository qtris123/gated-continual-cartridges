#!/usr/bin/env bash
# MECH-SEED (W3 BUILD + W5 VERIFY) — the confirmation standard, PLAN_AM_MUST_WIN.md §3.
#
# PART A (default preservation, two ways):
#   A1 snapshot     : `git archive HEAD` code, keys+reposition arm       -> the reference bytes
#   A2 edited/unset : working tree (MECH-007 built), AM_SEED_OFFSET unset
#   A3 edited/zero  : working tree, AM_SEED_OFFSET=0 explicitly
#   A1 == A2 == A3 byte-for-byte on all 16 per-document snapshots + cache_last.pt, else FAIL.
#
# PART B (the deliverable): the mission's best point re-run at TWO NON-ZERO seed offsets,
#   plus the mechanism-off control (KEY_MODE=freeze, no reposition) at both offsets, all
#   evaluated on BOTH splits at k=12 (the arm's minimum) and k=16 (its endpoint).
#
# Offset-0 gate: A2 must reproduce QA 2.034916400909424 / MT 2.330503463745117 at k=16 and
#   QA 1.9560121297836304 / MT 2.2720184326171875 at k=12 (DIAG-KEYCURVE).
#
# No import pin for the working-tree runs — this worker IS the source editor and the runs
# must see its edit. A1 is pinned to /tmp/amsnap_mechseed (git archive HEAD) by construction.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_mechseed
PY="$REPO/.venv/bin/python"

RESDIR="$REPO/research_loop/results/MECH-SEED"
LOGDIR="$RESDIR/evals"
RUNLOG="$RESDIR/runs"
mkdir -p "$LOGDIR" "$RUNLOG"
exec > "$RESDIR/wrapper.log" 2>&1

cd "$REPO" || exit 9

export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0

echo "MS_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" diff --stat -- cartridges examples > "$RESDIR/worktree_diff_stat.txt" 2>&1
git -C "$REPO" diff -- cartridges examples > "$RESDIR/worktree_diff.patch" 2>&1
echo "WORKTREE_DIFF_STAT:"; cat "$RESDIR/worktree_diff_stat.txt"

# ---- frozen HEAD snapshot for run A1 -----------------------------------------
rm -rf "$SNAP"; mkdir -p "$SNAP"
git -C "$REPO" archive HEAD cartridges examples | tar -x -C "$SNAP"
ln -sfn "$REPO/.venv" "$SNAP/.venv"
ln -sfn "$REPO/data"  "$SNAP/data"
manifest() { (cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort \
  | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1); }
echo "SNAPSHOT_PATH=$SNAP"
echo "SNAPSHOT_MANIFEST_AT_LAUNCH=$(manifest)"
echo "SNAPSHOT_HAS_SEED_OFFSET=$(grep -c 'seed_offset' "$SNAP/cartridges/am/finetune.py")"
echo "WORKTREE_HAS_SEED_OFFSET=$(grep -c 'seed_offset' "$REPO/cartridges/am/finetune.py")"

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
  echo "MS_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then echo "MS_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MS_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

SUMMARY="$RESDIR/curve.tsv"
: > "$SUMMARY"
printf "arm\toffset\tk\tsplit\tloss\tckpt\twandb_url\n" >> "$SUMMARY"
RUNDIRS="$RESDIR/rundirs.tsv"
: > "$RUNDIRS"

# =============================================================================
# one AM Phase-2 run.  $1 tag  $2 code_root  $3 key_mode  $4 reposition(1|-)
#                      $5 seed_offset(int|-)  $6 run_name
# =============================================================================
run_arm() {
  local tag="$1" root="$2" kmode="$3" repos="$4" soff="$5" runname="$6"
  local log="$RUNLOG/${tag}.log"
  echo "=== RUN ${tag} START $(date -Is) root=${root} key_mode=${kmode} repos=${repos} seed_offset=${soff}"
  local t0=$(date +%s)
  (
    export CARTRIDGES_DIR="$root"
    export PYTHONPATH="$root:${PYTHONPATH:-}"
    export AM_ROPE_THETA=5000000
    export MAX_QUERIES_PER_HEAD=64
    export SLOT_SELECTION=tfidf
    [ "$repos" = "-" ] || export AM_KEY_REPOSITION="$repos"
    [ "$soff"  = "-" ] || export AM_SEED_OFFSET="$soff"
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
    WANDB_GROUP=VERIFY \
    WANDB_NOTES="seed-varied confirmation of the mission's best point" \
    "$PY" "$root/examples/qasper2/train/continual_am_sparse.py" >"$log" 2>&1
  )
  local rc=$?
  local t1=$(date +%s)
  local dir
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  echo "RUN ${tag} RC=$rc WALL_S=$((t1-t0)) DIR=$dir"
  echo "RUN ${tag} IMPORT=$(grep -oE '/(tmp/amsnap_mechseed|localhome/local-triv/gated-continual-cartridges_explore)/cartridges' "$log" | head -1)"
  echo "RUN ${tag} SEEDLOG=$(grep -oE 'MECH-007:[^\"]*' "$log" | head -2 | tr '\n' ' ')"
  echo "RUN ${tag} WANDB=$(grep -oE 'https://wandb.ai/\S+/runs/[A-Za-z0-9]+' "$log" | tail -1)"
  echo "RUN ${tag} SUMMARY=$(tr -d '\n ' < "$dir/phase2_summary.json" 2>/dev/null)"
  printf "%s\t%s\t%s\t%s\n" "$tag" "$dir" "$rc" "$((t1-t0))" >> "$RUNDIRS"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

ckpt_for() {  # run_dir k -> cache-after-doc-<k-1>-*.pt
  local dir="$1" k="$2"
  ls "$dir"/cache-after-doc-$(printf "%03d" $((k-1)))-*.pt 2>/dev/null | head -1
}

hashes_of() {  # run_dir -> "<sha>  <basename>" for every cartridge artefact, sorted
  local dir="$1" f
  (cd "$dir" && for f in $(ls cache-after-doc-*.pt cache_last.pt 2>/dev/null | LC_ALL=C sort); do
      [ -r "$f" ] && sha256sum "$f" | awk '{print $1"  "$2}'
    done)
}

eval_one() {  # arm offset k split ckpt
  local arm="$1" off="$2" k="$3" split="$4" ckpt="$5"
  local log="$LOGDIR/eval_${arm}_off${off}_doc${k}_${split}.log"
  : > "$log"
  echo "--- [$(date -Is)] EVAL arm=${arm} off=${off} k=${k} split=${split} ckpt=${ckpt}"
  (
    export CARTRIDGES_DIR="$REPO"
    export PYTHONPATH="$REPO:${PYTHONPATH:-}"
    CHECKPOINT_PATH="$ckpt" \
    EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    RUN_NAME="MECH-SEED_off${off}_${arm}_k${k}_${split}" \
    WANDB_GROUP=VERIFY \
    WANDB_NOTES="seed-varied confirmation of the mission's best point" \
    WANDB_DISABLED=0 \
    "$PY" "$REPO/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1
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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$off" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK §1: eval_forgetting.py HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

eval_k12_k16() {  # arm offset run_dir
  local arm="$1" off="$2" dir="$3"
  local K C
  for K in 12 16; do
    C=$(ckpt_for "$dir" "$K")
    if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=$arm off=$off k=$K dir=$dir"; continue; fi
    eval_one "$arm" "$off" "$K" QA "$C"
    eval_one "$arm" "$off" "$K" MT "$C"
  done
}

# =============================================================================
# PART A — default preservation
# =============================================================================
echo "############ PART A: DEFAULT PRESERVATION ############"
run_arm A1_snapshot_head "$SNAP" highest_attention 1 -    MECH-SEED_off0_keys-snapshotHEAD ; RC_A1=$?
run_arm A2_edited_unset  "$REPO" highest_attention 1 -    MECH-SEED_off0_keys-flagunset    ; RC_A2=$?
run_arm A3_edited_zero   "$REPO" highest_attention 1 0    MECH-SEED_off0_keys-flagzero     ; RC_A3=$?

D_A1=$(cat "$RESDIR/rundir_A1_snapshot_head.txt"); D_A2=$(cat "$RESDIR/rundir_A2_edited_unset.txt")
D_A3=$(cat "$RESDIR/rundir_A3_edited_zero.txt")
hashes_of "$D_A1" > "$RESDIR/hashes_A1.txt"
hashes_of "$D_A2" > "$RESDIR/hashes_A2.txt"
hashes_of "$D_A3" > "$RESDIR/hashes_A3.txt"
echo "N_ARTEFACTS_A1=$(wc -l < "$RESDIR/hashes_A1.txt")"
if diff -q "$RESDIR/hashes_A1.txt" "$RESDIR/hashes_A2.txt" >/dev/null; then BITID_UNSET=yes; else BITID_UNSET=no; fi
if diff -q "$RESDIR/hashes_A1.txt" "$RESDIR/hashes_A3.txt" >/dev/null; then BITID_ZERO=yes; else BITID_ZERO=no; fi
echo "BIT_IDENTICAL_FLAG_UNSET_VS_HEAD_SNAPSHOT=$BITID_UNSET"
echo "BIT_IDENTICAL_FLAG_ZERO_VS_HEAD_SNAPSHOT=$BITID_ZERO"
diff "$RESDIR/hashes_A1.txt" "$RESDIR/hashes_A2.txt" > "$RESDIR/hashdiff_A1_A2.txt" 2>&1
diff "$RESDIR/hashes_A1.txt" "$RESDIR/hashes_A3.txt" > "$RESDIR/hashdiff_A1_A3.txt" 2>&1
# config.yaml diff: the new field appears with its default -> documented, not hidden
diff <(grep -v -E "run_dir|output_dir|^name:|launch_id|run_id|wandb|notes" "$D_A1/config.yaml") \
     <(grep -v -E "run_dir|output_dir|^name:|launch_id|run_id|wandb|notes" "$D_A2/config.yaml") \
     > "$RESDIR/config_diff_A1_A2.txt" 2>&1
echo "CONFIG_DIFF_A1_A2:"; cat "$RESDIR/config_diff_A1_A2.txt"

# =============================================================================
# OFFSET-0 GATE
# =============================================================================
echo "############ OFFSET-0 GATE (A2, flag unset) ############"
eval_k12_k16 keys 0 "$D_A2"
GATE=$("$PY" - "$SUMMARY" <<'EOF'
import sys
want = {(12,"QA"):1.9560121297836304, (12,"MT"):2.2720184326171875,
        (16,"QA"):2.034916400909424, (16,"MT"):2.330503463745117}
got, bad = {}, []
for line in open(sys.argv[1]).read().splitlines()[1:]:
    arm, off, k, sp, loss = line.split("\t")[:5]
    if arm == "keys" and off == "0" and loss != "MISSING":
        got[(int(k), sp)] = float(loss)
for key, w in want.items():
    g = got.get(key)
    if g is None or abs(g - w) > 1e-6:
        bad.append(f"{key}: got {g} want {w}")
print("GATE_FAIL " + " | ".join(bad) if bad else "GATE_PASS")
EOF
)
echo "OFFSET0_GATE=$GATE"
if [ "${GATE:0:9}" = "GATE_FAIL" ]; then
  echo "MS_ABORT_GATE_FAILED"
  cat "$SUMMARY"
  exit 4
fi

# =============================================================================
# PART B — non-zero offsets, keys arm + mechanism-off control
# =============================================================================
echo "############ PART B: SEED-VARIED ARMS ############"
run_arm B1_keys_off1000    "$REPO" highest_attention 1 1000 MECH-SEED_off1000_keys    ; RC_B1=$?
D_B1=$(cat "$RESDIR/rundir_B1_keys_off1000.txt"); eval_k12_k16 keys 1000 "$D_B1"

run_arm B2_keys_off2000    "$REPO" highest_attention 1 2000 MECH-SEED_off2000_keys    ; RC_B2=$?
D_B2=$(cat "$RESDIR/rundir_B2_keys_off2000.txt"); eval_k12_k16 keys 2000 "$D_B2"

run_arm B3_control_off1000 "$REPO" freeze            - 1000 MECH-SEED_off1000_control ; RC_B3=$?
D_B3=$(cat "$RESDIR/rundir_B3_control_off1000.txt"); eval_k12_k16 control 1000 "$D_B3"

run_arm B4_control_off2000 "$REPO" freeze            - 2000 MECH-SEED_off2000_control ; RC_B4=$?
D_B4=$(cat "$RESDIR/rundir_B4_control_off2000.txt"); eval_k12_k16 control 2000 "$D_B4"

# =============================================================================
# PART C — same-session offset-0 control + Phase-1 floor anchor
#   (existing MECH-KEYS control checkpoints; nothing re-solved)
# =============================================================================
echo "############ PART C: OFFSET-0 CONTROL + PHASE-1 ANCHOR (same session) ############"
D_C0="$REPO/outputs/2026-07-28-23-19-56-continual_am_sparse/508b8b0f-075a-429b-b757-2b11bfb57124"
if [ -d "$D_C0" ]; then eval_k12_k16 control 0 "$D_C0"; else echo "MISSING_MECHKEYS_CONTROL_DIR=$D_C0"; fi
eval_one phase1 0 0 QA "$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
eval_one phase1 0 0 MT "$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"

echo "MS_RC A1=$RC_A1 A2=$RC_A2 A3=$RC_A3 B1=$RC_B1 B2=$RC_B2 B3=$RC_B3 B4=$RC_B4"
echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
echo "MS_DONE_TS=$(date -Is)"
cat "$SUMMARY"
