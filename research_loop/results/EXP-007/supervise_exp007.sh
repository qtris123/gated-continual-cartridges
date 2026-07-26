#!/usr/bin/env bash
# Supervisor for EXP-007: both GPUs were busy at launch time (REF-ICL on gpu0,
# a concurrent AM-sparse run on gpu1). Per RUNBOOK §5 / EXECUTORS §A we WAIT and
# retry (never force). Relaunch the flock-claiming wrapper until it actually
# grabs a GPU (wrapper exits 3 == NO_FREE_GPU / lost the race). Cap the wait so
# this cannot spin forever.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
RESDIR="$PWD/research_loop/results/EXP-007"
SUPLOG="$RESDIR/supervisor.log"
: > "$SUPLOG"
MAX_WAIT_S=5400          # 90 min hard cap on waiting for a GPU
POLL_S=30
start=$(date +%s)
attempt=0
while : ; do
  attempt=$((attempt+1))
  echo "SUP_ATTEMPT=$attempt TS=$(date -Is)" >> "$SUPLOG"
  bash "$RESDIR/launch_exp007.sh"
  rc=$?
  echo "SUP_ATTEMPT=$attempt WRAPPER_RC=$rc TS=$(date -Is)" >> "$SUPLOG"
  if [ "$rc" -ne 3 ]; then
    echo "SUP_DONE rc=$rc (wrapper claimed a GPU and ran, or a real failure) TS=$(date -Is)" >> "$SUPLOG"
    exit "$rc"
  fi
  now=$(date +%s)
  if [ $((now - start)) -ge "$MAX_WAIT_S" ]; then
    echo "SUP_GAVE_UP: no GPU freed within ${MAX_WAIT_S}s TS=$(date -Is)" >> "$SUPLOG"
    exit 3
  fi
  sleep "$POLL_S"
done
