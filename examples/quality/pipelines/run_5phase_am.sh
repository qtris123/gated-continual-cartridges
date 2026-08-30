#!/usr/bin/env bash
# QuALITY: classic P1 AM compaction, then one canonical AM write per P2-P5 story.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PY="${CARTRIDGES_PYTHON:-${PY:-$ROOT/.venv/bin/python}}"
RUNS="$ROOT/outputs/quality_5phase_runs"
STATE="$ROOT/outputs/quality_5phase_state"
GPU="${GPU:-3}"

export CARTRIDGES_DIR="$ROOT"
export CARTRIDGES_OUTPUT_DIR="$RUNS"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

mkdir -p "$RUNS" "$STATE"
for phase in 1 2 3 4 5; do
  eval_path="$ROOT/data/quality/phases/phase${phase}_eval.parquet"
  synth_path="$ROOT/data/quality/train/qwen_quality_p${phase}_task_8192.parquet"
  [ -f "$eval_path" ] || { echo "missing $eval_path" >&2; exit 2; }
  [ -f "$synth_path" ] || { echo "missing $synth_path" >&2; exit 2; }
  export "EVAL_P${phase}_PATH=$eval_path"
  export "EVAL_P${phase}_NAME=quality_p${phase}"
done

marker_cache() {
  "$PY" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text())
cache = data.get("cache_path")
expected = {f"quality_p{i}" for i in range(1, 6)}
if not cache or not Path(cache).exists() or not expected <= set(data.get("eval_metrics", {})):
    raise SystemExit(1)
print(cache)
PY
}

record_latest() {
  local pattern="$1" marker="$2" started="$3"
  "$PY" - "$RUNS" "$pattern" "$marker" "$started" <<'PY'
import glob
import json
import sys
from pathlib import Path

runs, pattern, marker, started = sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
candidates = [
    Path(path)
    for path in glob.glob(str(Path(runs) / pattern))
    if Path(path).stat().st_mtime >= started - 2
]
if not candidates:
    raise SystemExit(f"no completed summary matching {pattern}")
summary_path = max(candidates, key=lambda path: path.stat().st_mtime)
data = json.loads(summary_path.read_text())
cache = (summary_path.parent / "cache_last.pt").resolve()
data["cache_path"] = str(cache)
expected = {f"quality_p{i}" for i in range(1, 6)}
if not cache.exists() or not expected <= set(data.get("eval_metrics", {})):
    raise SystemExit(f"incomplete run summary {summary_path}")
phase = int(marker.stem[1:]) if marker.stem.startswith("p") else 1
sys.path.insert(0, str(Path(runs).parent))
from examples.shared.evaluate.cache_layout import publish_cache

published = publish_cache(
    cache,
    "quality",
    phase,
    "am-canonical-512",
    metadata={"run_name": data.get("run_name"), "state_path": str(marker)},
)
data["original_cache_path"] = str(cache)
data["cache_path"] = str(published)
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(json.dumps(data, indent=2))
print(published)
PY
}

echo "=== QuALITY AM chain | GPU $GPU | $(date) ==="

P1_MARKER="$STATE/p1.json"
if P1_CACHE="$(marker_cache "$P1_MARKER" 2>/dev/null)"; then
  echo "P1 complete: $P1_CACHE"
else
  started="$(date +%s)"
  AM_DATASET=quality \
  AM_QUALITY_PHASE=1 \
  QA_DATA_PATH="$ROOT/data/quality/train/qwen_quality_p1_task_8192.parquet" \
  NUM_TOKENS="${NUM_TOKENS:-512}" \
  KEY_SELECT="${KEY_SELECT:-highest_attention}" \
  ENABLE_BETA="${ENABLE_BETA:-0}" \
  RIDGE_LAMBDA="${P1_RIDGE_LAMBDA:-1e-4}" \
  RIDGE_SCALE="${RIDGE_SCALE:-spectral}" \
  GRANULARITY="${P1_GRANULARITY:-per_head}" \
  REBAKE_KEY_POSITIONS="${REBAKE_KEY_POSITIONS:-1}" \
  AM_ROPE_THETA="${AM_ROPE_THETA:-model}" \
  GLOBAL_TEACHER_POSITIONS="${GLOBAL_TEACHER_POSITIONS:-1}" \
  RUN_NAME="QUALITY_P1_AM_COMPACTION" \
  "$PY" "$ROOT/examples/shared/am/initial_compaction.py"
  P1_CACHE="$(record_latest '*-initial_am_compaction/*/summary.json' "$P1_MARKER" "$started")"
fi

input_cache="$P1_CACHE"
for phase in 2 3 4 5; do
  marker="$STATE/p${phase}.json"
  if output_cache="$(marker_cache "$marker" 2>/dev/null)"; then
    echo "P$phase complete: $output_cache"
    input_cache="$output_cache"
    continue
  fi

  run_name="QUALITY_P${phase}_delta_ha_b0_idf0"
  started="$(date +%s)"
  PHASE1_CACHE_PATH="$input_cache" \
  SYNTH_DATA_PATH="$ROOT/data/quality/train/qwen_quality_p${phase}_task_8192.parquet" \
  AM_DATASET=quality \
  AM_QUALITY_PHASE="$phase" \
  AM_ROPE_THETA=model \
  TOP_T="${TOP_T:-32}" \
  GRANULARITY="${GRANULARITY:-per_layer}" \
  SLOT_SELECTION=tfidf \
  USE_IDF=0 \
  KEY_MODE=highest_attention \
  AM_KEY_REPOSITION=1 \
  ENABLE_BETA=0 \
  RIDGE_LAMBDA=0 \
  RIDGE_LAMBDA_MIN=0 \
  DELTA_WEIGHT="${DELTA_WEIGHT:-1e-2}" \
  SAVE_AFTER_EACH_DOCUMENT=1 \
  RUN_NAME="$run_name" \
  "$PY" "$ROOT/examples/shared/am/continual_write.py"

  output_cache="$(
    record_latest "*-${run_name}/*/phase2_summary.json" "$marker" "$started"
  )"
  input_cache="$output_cache"
done

echo "=== QuALITY AM chain complete: $input_cache | $(date) ==="
