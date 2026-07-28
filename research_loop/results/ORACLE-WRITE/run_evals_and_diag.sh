#!/usr/bin/env bash
# ORACLE-WRITE part 2: independent eval_forgetting.py on BOTH checkpoints x BOTH splits,
# then the B-ROUTE eval-time attention-mass diagnostic. One flock-claimed GPU.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
PY="$CARTRIDGES_DIR/.venv/bin/python"

RESDIR="$CARTRIDGES_DIR/research_loop/results/ORACLE-WRITE"
exec > "$RESDIR/wrapper_evals.log" 2>&1

CTRL_DIR="$CARTRIDGES_DIR/outputs/2026-07-28-19-27-08-continual_am_sparse/43a46f4b-5341-4a0c-98f3-2305daa8247f"
ORAC_DIR="$CARTRIDGES_DIR/outputs/2026-07-28-19-30-39-continual_am_sparse/021eb2f7-253b-4649-b397-892f31c7d12b"
PHASE1="$CARTRIDGES_DIR/outputs/phase1_selfdistill_qwen512/cache_last.pt"

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
CLAIMED=""
for idx in 1 0; do
  exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
  if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
done
[ -z "$CLAIMED" ] && { echo "NO_FREE_GPU"; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
echo "EVALS_CLAIMED_GPU=$CLAIMED"
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

eval_one() {  # label ckpt split
  local label="$1" ckpt="$2" split="$3"
  local log="$RESDIR/eval_${label}_${split}.log"
  : > "$log"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="ORACLE-WRITE_${label}_eval${split}" \
  WANDB_GROUP=B-ROUTE \
  WANDB_DISABLED=0 \
  "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "RESULT ${label} ${split}: $(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1)  ckpt=$ckpt"
  echo "WANDB  ${label} ${split}: $(grep -oE 'https://wandb.ai/\S+/runs/\S+' "$log" | tail -1)"
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

for SPLIT in QA MT; do
  eval_one control "$CTRL_DIR/cache_last.pt" "$SPLIT"
  eval_one oracle  "$ORAC_DIR/cache_last.pt" "$SPLIT"
  eval_one phase1  "$PHASE1"                 "$SPLIT"
done

echo "=== EVALS DONE, starting route-mass diagnostic ==="
CACHE_SPECS="phase1_oracleS:$PHASE1:$ORAC_DIR,oracle:$ORAC_DIR/cache_last.pt:-,phase1_ctrlS:$PHASE1:$CTRL_DIR,control:$CTRL_DIR/cache_last.pt:-" \
EVAL_SPECS="QA:data/qasper/eval/qasper_eval_QA.parquet,MT:data/qasper/eval/qasper_eval_MT.parquet" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/ORACLE-WRITE_route_mass.json" \
"$PY" "$RESDIR/measure_route_mass.py"
echo "DIAG_RC=$?"
echo "ALL_DONE_TS=$(date -Is)"
