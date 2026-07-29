#!/usr/bin/env bash
# DIAG-KEYCURVE (B-ROUTE key side / B-OVERWRITE)
#
# PART 1 (primary): the per-document acquisition curve of MECH-KEYS' BEST arm
#   (KEY_MODE=highest_attention + AM_KEY_REPOSITION=1, theta=5e6, beta off, top32).
#   MECH-KEYS saved 16 `cache-after-doc-*.pt` snapshots for that arm and never evaluated
#   them. The value-only curve bottoms at k=12 (MT 2.4352) and then DEGRADES to 2.5524
#   by k=16, so the keys arm's reported k=16 endpoint may not be its best point.
#   Pure eval of existing artefacts: no training, no instrumentation, no source edit.
#   Correctness gate: k=16 must reproduce QA 2.0349 / MT 2.3305.
#
# PART 2 (W5 verify): fresh-process re-run of the keys+reposition arm, plus a
#   same-session re-eval of the mechanism-off control (KEY_MODE=freeze) checkpoints.
#
# PART 3: eval-time mass_on_S / MT-QA mass ratio / total cartridge mass at every k
#   (per-k slot set = union of the first k documents' selections).
#
# Imports pinned to a frozen `git archive HEAD` snapshot (RUNBOOK 9c-bis) because the
# D0-ICL worker is live on the other GPU.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_kcurve
PY="$REPO/.venv/bin/python"

export CARTRIDGES_DIR="$SNAP"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"

RESDIR="$REPO/research_loop/results/DIAG-KEYCURVE"
LOGDIR="$RESDIR/evals"
mkdir -p "$LOGDIR"
exec > "$RESDIR/wrapper.log" 2>&1

cd "$REPO" || exit 9

# the pinned driver needs a venv + data under CARTRIDGES_DIR (git archive ships neither)
ln -sfn "$REPO/.venv" "$SNAP/.venv"
ln -sfn "$REPO/data"  "$SNAP/data"

manifest() { (cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort \
  | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1); }

echo "KC_START_TS=$(date -Is)"
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
  echo "KC_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then echo "KC_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "KC_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

# ---- MANDATORY: prove the pin, from /tmp (NOT the repo root) -----------------
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

DIR_R="$REPO/outputs/2026-07-28-23-37-22-continual_am_sparse/a20bc87b-e681-4122-90ed-24b3495f6849"  # MECH-KEYS keys_repos
DIR_C="$REPO/outputs/2026-07-28-23-19-56-continual_am_sparse/508b8b0f-075a-429b-b757-2b11bfb57124"  # MECH-KEYS control (freeze)
echo "KC_DIR_keys_repos=$DIR_R"
echo "KC_DIR_control=$DIR_C"

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
  RUN_NAME="DIAG-KEYCURVE_${arm}_doc${k}_${split}" \
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
# PART 1 -- keys+reposition k-curve. Brief's k first (16 is the gate),
# then the fill-in k so the shape is directly comparable to DIAG-SEQUENCE.
# =====================================================================
echo "=== PART 1: keys_repos k-curve (brief priority) ==="
for K in 16 12 14 10 8 4 15 6; do
  C=$(ckpt_for "$DIR_R" "$K")
  if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=keys_repos k=$K"; continue; fi
  eval_one keys_repos "$K" QA "$C"
  eval_one keys_repos "$K" MT "$C"
done

# ---- locate the MT minimum so far; the control is evaluated there too --------
ARGMIN_K=$("$PY" - "$SUMMARY" <<'EOF'
import sys
best_k, best = None, float("inf")
for line in open(sys.argv[1]).read().splitlines()[1:]:
    a, k, sp, loss = line.split("\t")[:4]
    if a == "keys_repos" and sp == "MT" and loss != "MISSING":
        v = float(loss)
        if v < best: best, best_k = v, int(k)
print(best_k if best_k is not None else 16)
EOF
)
echo "KC_KEYS_ARGMIN_MT_K=$ARGMIN_K"

# =====================================================================
# PART 2a -- mechanism-OFF control (KEY_MODE=freeze), same session, same harness.
# =====================================================================
echo "=== PART 2a: control (freeze) curve at the comparison k ==="
CTRL_KS=$(printf "%s\n" 16 12 14 8 "$ARGMIN_K" | sort -rn | uniq)
for K in $CTRL_KS; do
  C=$(ckpt_for "$DIR_C" "$K")
  if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=control k=$K"; continue; fi
  eval_one control "$K" QA "$C"
  eval_one control "$K" MT "$C"
done

# =====================================================================
# PART 2b -- fresh-process RE-RUN of the keys+reposition arm.
#   No SEED knob exists (see seed_probe.json: `pydrantic.main` is only reached in
#   AM_EXECUTION_MODE=train_loop, so `seed=<n>` never reaches config.seed; and the
#   per-document reference draw is seeded by doc_idx, not config.seed). So this is a
#   same-seed, fresh-process, different-GPU-context reproduction and NOTHING MORE.
# =====================================================================
echo "=== PART 2b: fresh-process re-run of keys+reposition ==="
RERUN_LOG="$RESDIR/rerun_keys_repos.log"
echo "KC_RERUN_START_EPOCH=$(date +%s) TS=$(date -Is)"
# The driver is invoked DIRECTLY with the pinned interpreter (not via
# train_continual_am_sparse.sh) so that the venv is the repo's real one while the code is
# the frozen snapshot. Every var the wrapper would have injected is set here explicitly;
# the ones omitted (EPOCHS/GLOBAL_BATCH_SIZE/MAX_STEPS/UPDATE_INTERVAL/QUERIES_PER_BATCH/
# DISTRIBUTED_BACKEND/OLD_*) have identical defaults in the wrapper and in the driver.
(
  export TORCH_CUDA_ARCH_LIST=9.0
  export AM_ROPE_THETA=5000000
  export AM_KEY_REPOSITION=1
  export MAX_QUERIES_PER_HEAD=64
  export SLOT_SELECTION=tfidf
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
  RUN_NAME=DIAG-KEYCURVE_rerun_keys-repos \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-ROUTE \
  WANDB_NOTES="W5 fresh-process reproduction of MECH-KEYS keys+reposition (QA 2.0349 / MT 2.3305); no seed knob exists" \
  "$PY" "$SNAP/examples/qasper2/train/continual_am_sparse.py" >"$RERUN_LOG" 2>&1
)
RC_RERUN=$?
echo "KC_RERUN_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$RC_RERUN"
DIR_RR=$(grep -oE "Saved to /localhome/\S+" "$RERUN_LOG" | tail -1 | awk '{print $3}')
echo "KC_RERUN_RUNDIR=$DIR_RR"
echo "KC_RERUN_WANDB=$(grep -oE 'https://wandb.ai/\S+/runs/[a-zA-Z0-9]+' "$RERUN_LOG" | tail -1)"
echo "KC_RERUN_SUMMARY=$(cat "$DIR_RR/phase2_summary.json" 2>/dev/null | tr -d '\n ')"
echo "$DIR_RR" > "$RESDIR/rundir_rerun.txt"

# config.yaml diff against the original arm: proves the re-run is the same configuration
if [ -n "$DIR_RR" ]; then
  diff <(grep -v -E "run_dir|output_dir|^name:|wandb" "$DIR_R/config.yaml") \
       <(grep -v -E "run_dir|output_dir|^name:|wandb" "$DIR_RR/config.yaml") \
       > "$RESDIR/config_diff.txt" 2>&1
  echo "KC_CONFIG_DIFF_LINES=$(wc -l < "$RESDIR/config_diff.txt")"
fi

# =====================================================================
# PART 2c -- standalone evals of the re-run's final cartridge + its own k-curve
#            at the two decisive k (the k=16 endpoint and the arm's argmin).
# =====================================================================
echo "=== PART 2c: standalone evals of the re-run ==="
if [ -n "$DIR_RR" ]; then
  eval_one rerun 16 QA "$DIR_RR/cache_last.pt"
  eval_one rerun 16 MT "$DIR_RR/cache_last.pt"
  if [ "$ARGMIN_K" != "16" ]; then
    C=$(ckpt_for "$DIR_RR" "$ARGMIN_K")
    if [ -n "$C" ]; then
      eval_one rerun "$ARGMIN_K" QA "$C"
      eval_one rerun "$ARGMIN_K" MT "$C"
    fi
  fi
fi

# =====================================================================
# PART 1b -- fill in the remaining k for the keys_repos arm (shape comparison).
# =====================================================================
echo "=== PART 1b: keys_repos k-curve fill-in ==="
for K in 13 11 9 7 5 3 2 1; do
  C=$(ckpt_for "$DIR_R" "$K")
  if [ -z "$C" ]; then echo "MISSING_SNAPSHOT arm=keys_repos k=$K"; continue; fi
  eval_one keys_repos "$K" QA "$C"
  eval_one keys_repos "$K" MT "$C"
done

# k=0 anchor (untouched Phase-1 cartridge), same session
eval_one phase1 0 QA "$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
eval_one phase1 0 MT "$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"

# =====================================================================
# PART 3 -- eval-time mass_on_S / MT-QA ratio / cartridge mass at every k.
#   Per-k slot set S = union of the first k documents' per-layer top-32 selections,
#   built as a symlink dir holding only am_doc_doc-000..doc-(k-1) (no code change).
# =====================================================================
echo "=== PART 3: route-mass per k ==="
SLOTDIR=/tmp/kcurve_slots
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
  C=$(ckpt_for "$DIR_R" "$K"); [ -z "$C" ] && continue
  D=$(build_slotdir "$DIR_R" "$K" keys_repos)
  SPECS="${SPECS}keys_repos_k${K}:$C:$D,"
done
for K in $CTRL_KS; do
  C=$(ckpt_for "$DIR_C" "$K"); [ -z "$C" ] && continue
  D=$(build_slotdir "$DIR_C" "$K" control)
  SPECS="${SPECS}control_k${K}:$C:$D,"
done
if [ -n "$DIR_RR" ]; then
  D=$(build_slotdir "$DIR_RR" 16 rerun)
  SPECS="${SPECS}rerun_k16:$DIR_RR/cache_last.pt:$D,"
fi
# Phase-1 reference measured on the CONTROL's full 16-document union, exactly as
# MECH-KEYS did -> its 0.08778 (MT) / 0.08119 (QA) is a cross-session reproduction check.
SPECS="${SPECS}phase1:$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt:$DIR_C"
echo "KC_CACHE_SPECS=$SPECS"
cp "$REPO/research_loop/results/MECH-KEYS/measure_route_mass.py" "$RESDIR/measure_route_mass.py"
echo "KC_ROUTEMASS_SCRIPT_SHA256=$(sha256sum "$RESDIR/measure_route_mass.py" | cut -d' ' -f1)"
CACHE_SPECS="$SPECS" \
EVAL_SPECS="QA:$REPO/data/qasper/eval/qasper_eval_QA.parquet,MT:$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
OUT_JSON="$REPO/research_loop/state/diagnostics/DIAG-KEYCURVE_route_mass.json" \
"$PY" "$RESDIR/measure_route_mass.py" > "$RESDIR/route_mass.log" 2>&1
echo "KC_ROUTE_MASS_RC=$?"
tail -40 "$RESDIR/route_mass.log"

echo "KC_DONE_TS=$(date -Is)"
echo "SNAPSHOT_MANIFEST_AFTER=$(manifest)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
cat "$SUMMARY"
