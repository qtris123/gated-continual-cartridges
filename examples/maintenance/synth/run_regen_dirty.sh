#!/usr/bin/env bash
# After deleting quarantined checkpoints, regenerate FinQA p03-p05 and TechQA p02-p03.
set -euo pipefail

cd /localhome/local-triv/gated-continual-cartridges
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export PYTHONPATH=$CARTRIDGES_DIR
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

VLLM_VENV="${VLLM_VENV:-$CARTRIDGES_DIR/.venv-vllm}"
AM_PY="${AM_PY:-$VLLM_VENV/bin/python}"
export AM_PY

TORCH_LIB="$("$AM_PY" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "=============================================================="
echo "start: $(date)"
echo "CARTRIDGES_DIR=$CARTRIDGES_DIR"
echo "AM_PY=$AM_PY"
"$AM_PY" -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print('vllm', vllm.__version__)"
echo "=============================================================="

echo
echo "=== FinQA p03 p04 p05 (fresh) ==="
PHASES="3 4 5" DP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/finqa/synthesize/self_study.sh

echo
echo "=== TechQA p02 p03 (fresh) ==="
HF_HUB_OFFLINE=0 PHASES="2 3" DP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/techqa/synthesize/self_study.sh

echo "=============================================================="
echo "finish: $(date)"
echo "=============================================================="
