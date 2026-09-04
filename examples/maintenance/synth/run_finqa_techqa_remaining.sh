#!/usr/bin/env bash
# Fill remaining TechQA p04-p05 and FinQA p03-p05 self-study artifacts.
set -euo pipefail

cd /localhome/local-triv/gated-continual-cartridges
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export PYTHONPATH=$CARTRIDGES_DIR
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# Dedicated env so we do not replace torch 2.7.1+cu128 in .venv (training).
VLLM_VENV="${VLLM_VENV:-$CARTRIDGES_DIR/.venv-vllm}"
if [ ! -x "$VLLM_VENV/bin/python" ]; then
  echo "creating $VLLM_VENV"
  /usr/bin/python3 -m venv "$VLLM_VENV"
fi
AM_PY="${AM_PY:-$VLLM_VENV/bin/python}"
export AM_PY

echo "=============================================================="
echo "start: $(date)"
echo "CARTRIDGES_DIR=$CARTRIDGES_DIR"
echo "AM_PY=$AM_PY"
echo "=============================================================="

if ! "$AM_PY" -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null; then
  echo "installing vllm==0.11.0 + synth client deps into $VLLM_VENV"
  "$AM_PY" -m pip install -U pip
  "$AM_PY" -m pip install 'vllm==0.11.0' 'pydrantic==0.0.3' wandb pyarrow pandas httpx
  "$AM_PY" -c "import vllm; print('vllm', vllm.__version__)"
fi
# PyPI torch 2.8.0 on aarch64 is CPU-only; cu128 wheels skip 2.8.0.
# Use the same 2.7.1+cu128 stack as .venv so libcudart.so.12 is present.
if ! "$AM_PY" -c "import torch; assert torch.cuda.is_available() and torch.version.cuda" 2>/dev/null; then
  echo "replacing CPU torch with 2.7.1+cu128 wheels"
  "$AM_PY" -m pip install --force-reinstall \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
fi
TORCH_LIB="$("$AM_PY" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
"$AM_PY" -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print('vllm', vllm.__version__)"

echo
echo "=== TechQA remaining stages p04 p05 ==="
# Dataset is nvidia/TechQA-RAG-Eval; it is not in the local Hub cache.
HF_HUB_OFFLINE=0 PHASES="4 5" DP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/techqa/synthesize/self_study.sh

echo
echo "=== FinQA quarantined stages p03 p04 p05 ==="
PHASES="3 4 5" DP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/finqa/synthesize/self_study.sh

echo "=============================================================="
echo "finish: $(date)"
echo "=============================================================="
