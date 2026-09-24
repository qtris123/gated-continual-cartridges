#!/usr/bin/env bash
# Four-GPU launcher for the Llama 3.2 3B 25% sweep in CHECKLIST_LLAMA_SWEEP.md.
# Ten independent chains (5 datasets x 2 arms). Writes stay on one GPU each.
# GPUs default to 2,3,4,5 so occupied devices 0 and 1 are left alone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPUS_STR="${GPUS:-2,3,4,5}"
LOGROOT="$ROOT/logs/e2e_subkv_sweep/llama_4gpu"
mkdir -p "$LOGROOT"

IFS=',' read -r -a GPUS <<< "$GPUS_STR"

# Heavy 8192 arms first so QuALITY and LongHealth start immediately.
JOBS=(
  "quality 8192 2048"
  "longhealth 8192 2048"
  "qasper 8192 2048"
  "techqa 8192 2048"
  "finqa 8192 2048"
  "quality 512 128"
  "longhealth 512 128"
  "qasper 512 128"
  "techqa 512 128"
  "finqa 512 128"
)

QUEUE="$LOGROOT/queue.txt"
LOCK="$LOGROOT/queue.lock"
# KEEP_QUEUE=1 attaches extra GPUs to a queue another launcher already filled.
if [[ "${KEEP_QUEUE:-0}" != "1" ]]; then
  : > "$QUEUE"
  : > "$LOCK"
  # ONLY_DATASETS=quality,finqa skips the other datasets. Used when a dataset
  # is already running on another GPU so a second launcher does not collide.
  ONLY="${ONLY_DATASETS:-}"
  for job in "${JOBS[@]}"; do
    if [[ -n "$ONLY" ]]; then
      ds="${job%% *}"
      case ",${ONLY}," in
        *",${ds},"*) ;;
        *) continue ;;
      esac
    fi
    echo "$job" >> "$QUEUE"
  done
fi

export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
export CARTRIDGES_PYTHON="${CARTRIDGES_PYTHON:-$ROOT/.venv/bin/python}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_DISABLED=1
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.config/gcc-hf-token" ]]; then
  HF_TOKEN="$(cat "$HOME/.config/gcc-hf-token")"
  export HF_TOKEN
fi

run_job() {
  local gpu="$1" dataset="$2" budget="$3" top_t="$4"
  local stamp logfile extra=()
  stamp="$(date +%Y%m%d_%H%M%S)"
  logfile="$LOGROOT/${dataset}_b${budget}_t${top_t}_gpu${gpu}_${stamp}.log"
  if [[ "$dataset" == "quality" || "$dataset" == "longhealth" ]]; then
    extra+=(--eval-accuracy)
  fi
  echo "START gpu=$gpu dataset=$dataset budget=$budget top_t=$top_t log=$logfile"
  set +e
  bash "$ROOT/examples/e2e_subkv_sweep/run_sweep_llama.sh" \
    --dataset "$dataset" \
    --budgets "$budget" \
    --top-ts "$top_t" \
    --gpu "$gpu" \
    --eval-gpus "$gpu" \
    "${extra[@]}" \
    >"$logfile" 2>&1
  local rc=$?
  set -e
  echo "END gpu=$gpu dataset=$dataset budget=$budget top_t=$top_t rc=$rc"
  return "$rc"
}

worker() {
  local gpu="$1"
  while true; do
    local job
    job="$(flock "$LOCK" bash -c 'line=$(head -n 1 "$1" || true); [[ -n "$line" ]] || exit 1; tail -n +2 "$1" > "$1.tmp" && mv "$1.tmp" "$1"; printf "%s\n" "$line"' _ "$QUEUE")" || break
    # Keep the GPU working after a failed chain; the log has the traceback.
    # shellcheck disable=SC2086
    run_job "$gpu" $job || echo "FAILED gpu=$gpu job=$job"
  done
}

pids=()
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo "llama 4gpu sweep finished fail=$fail"
exit "$fail"
