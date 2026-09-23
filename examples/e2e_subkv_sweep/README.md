# End-to-End Continual Compaction Sweep (Sub-KV Cache Sizes)

This directory contains the automated pipeline script and runbook for running an end-to-end continual KV-cache compaction sweep on a standalone GPU server (without Slurm or cluster queue managers).

The pipeline systematically sweeps over **sub-KV cache budgets** ($1024 \to 2048 \to 4096$, with $512$ available as reference) while scaling the selective update budget **$\text{top\_t}$ proportionally** ($64 \to 128 \to 256$):

| Budget ($S$) | Proportional $\text{top\_t}$ ($t$) | Ratio ($t / S$) | Stage 1 (Initial Compaction) | Stages 2–5 (Continual AM) | Status |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **512** | **32** | 6.25% | Initial Compaction ($N=512$) | Continual 5-stage chain ($t=32$) | Completed / Reference |
| **1024** | **64** | 6.25% | Initial Compaction ($N=1024$) | Continual 5-stage chain ($t=64$) | **Active Sweep** |
| **2048** | **128** | 6.25% | Initial Compaction ($N=2048$) | Continual 5-stage chain ($t=128$) | **Active Sweep** |
| **4096** | **256** | 6.25% | Initial Compaction ($N=4096$) | Continual 5-stage chain ($t=256$) | **Active Sweep** |
| **16384** | **1024, 4096, 8192** | 6.25%, 25%, 50% | Initial Compaction ($N=16384$) | Continual 5-stage chain ($t \in \{1024, 4096, 8192\}$) | **Top-t Capacity Sweep** |

Each arm executes the canonical experimental setting:
- **Stage 1 (Initial Compaction / p01):** Closed-form Arm-D compaction with spectral ridge regularization $\lambda_{\text{ridge}} = 10^{-4}$ (`p01.ridge_lambda: 1e-4`, `p01.ridge_scale: spectral`). This regularizes the underdetermined least-squares system ($N_{\text{queries}} \le 64 \ll S$) and ensures unqueried slot values decay stably rather than diverging.
- **Stages 2–5 (Continual AM / p02–p05):** Closed-form attention-output delta matching with trust-region weight $\lambda_\Delta = 0.01$ (`objective.delta_weight: 0.01`) and $\lambda_{\text{ridge}} = 0.0$ (the $\lambda_\Delta$ term acts as the $L_2$ anchor against prior values).
- **Key Selection:** Highest-attention key selection (`key_mode: highest_attention`).
- **Attention Bias:** No attention-bias fitting (`beta.enabled: false`, `enable_beta: 0`).
- **Positional Integrity:** Fixed RoPE repositioning (`key_reposition: true`, `rebake_key_positions: 1`, `rope_theta: 5e6`).
- **Multi-Stage Chain:** Phase 1 initial compaction ($S$ slots) followed by Phases 2–5 continual AM updates ($t$ selective writes per layer per document) and 5×5 held-out evaluations.

> [!NOTE]
> **Why Phase 1 requires $\lambda_{\text{ridge}} = 10^{-4}$ while Stages 2–5 use $\lambda_{\text{ridge}} = 0.0$:**
> In continual AM (Stages 2–5), the objective includes a quadratic penalty anchoring the new values to the previous stage's values ($\lambda_\Delta \|V_S - V_{S,\text{orig}}\|^2$). In Phase 1, there is no prior cartridge or $\lambda_\Delta$ anchor; setting `p01.ridge_lambda: 0` removes all regularization from `torch.linalg.lstsq(driver="gels")`, causing unconstrained slot values to fit noise and blow up. Setting $\lambda_{\text{ridge}} = 10^{-4}$ uses the dual Cholesky solve $W = X^T(XX^T + \lambda I)^{-1}Y$, keeping unqueried slots well-conditioned and stable.

---

## 1. Environment Setup

### System Requirements
- Linux OS (Ubuntu 20.04/22.04 or RHEL/CentOS/Rocky 8/9).
- NVIDIA GPU with at least 24 GB VRAM (e.g., RTX 3090/4090, A5000, A30, A100, A40, H100).
- Python 3.10, 3.11, or 3.12 with CUDA drivers installed.
- `zstd` utility installed (`sudo apt-get install zstd` or `sudo dnf install zstd`).

### Step 1: Clone and Create Python Environment

From the project root (`gated-continual-cartridges`):

```bash
# Option A: Using venv
python3 -m venv .venv
source .venv/bin/activate

# Option B: Using conda
conda create -n cartridges python=3.12 -y
conda activate cartridges
```

### Step 2: Install Dependencies

All dependencies are declared in [`pyproject.toml`](../../pyproject.toml). Install the package in editable mode:

```bash
# Standard pip install
pip install --upgrade pip
pip install -e .

# Or using uv (fast)
pip install uv
uv pip install -e .
```

Alternatively, if using Conda, you can recreate the full environment from the root [`environment.yml`](../../environment.yml):
```bash
conda env create -f environment.yml
conda activate cartridges
```

> [!IMPORTANT]
> **HuggingFace Hub Compatibility:**
> `transformers` requires `huggingface-hub >= 0.34.0, < 1.0`. Installing `huggingface-hub >= 1.0` will trigger an `ImportError`. This constraint is pinned in [`pyproject.toml`](../../pyproject.toml), ensuring `pip install -e .` resolves the compatible versions automatically.

### Step 3: Verify GPU & PyTorch

Check that PyTorch detects your GPU:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 2. Download Datasets & Checkpoints from Hugging Face

The long-context phase corpora, evaluation sets, and synthetic self-study datasets live in Hugging Face dataset repositories.

### Step 1: Set Hugging Face Token

Export your Hugging Face access token (needed for accessing model weights and dataset shards):

```bash
export HF_TOKEN="hf_your_token_here"
```

### Step 2: Hydrate Datasets into `data/`

Run the artifact preparation script to download and unpack all benchmark datasets:

```bash
# Download and unpack data/ (phase corpora, eval sets, synthetic training parquets)
bash scripts/prepare_artifacts.sh --which data
```

*(Optional: If you also want to download precomputed reference baseline results and figures, run `bash scripts/prepare_artifacts.sh --which results`).*

### Step 3: Verify Data Presence

Ensure that the synthetic train and held-out evaluation parquets exist:

```bash
python3 -c '
from examples.shared.am.continual_env import DATASETS, spec
for ds in ["qasper", "quality", "finqa", "techqa"]:
    s = spec(ds)
    all_ok = all(s.synth_path(p).exists() and s.eval_path(p).exists() for p in range(1, 6))
    print(f"[{'OK' if all_ok else 'MISSING'}] {ds}")
'
```

---

## 3. Running the Automated E2E Pipeline

The runner script is located at [`examples/e2e_subkv_sweep/run_sweep.sh`](run_sweep.sh).

### Quick Start (Default Run on GPU 0)

To run the sweep (1024, 2048, 4096) on `qasper`:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --gpu 0
```

*(Note: Budget 512 with top-t 32 has already been run, but if you want to include it in a 4-point sweep: `bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --budgets 512,1024,2048,4096 --top-ts 32,64,128,256 --gpu 0`)*

### Running with a Multi-GPU Setup
If your server has multiple GPUs, you can assign one GPU for closed-form writes and fan held-out evaluations across other GPUs:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh \
  --dataset qasper \
  --gpu 0 \
  --eval-gpus 0,1
```

### Customizing Budgets and Top-$t$
You can specify custom budgets and proportional top-$t$ ratios:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh \
  --dataset quality \
  --budgets 1024,2048 \
  --top-ts 64,128 \
  --gpu 0
```

### Sweeping Top-t for 16,384-Slot Budget

To sweep over selective write budgets $t \in \{1024, 4096, 8192\}$ for a fixed 16,384-slot sub-KV cache:

```bash
# Using the dedicated convenience wrapper (Qwen 3 4B on QASPER):
bash examples/e2e_subkv_sweep/run_sweep_16k.sh --gpu 0

# For Llama 3.2 3B:
bash examples/e2e_subkv_sweep/run_sweep_16k.sh --model meta-llama/Llama-3.2-3B-Instruct --gpu 0

# Or via run_sweep.sh directly:
bash examples/e2e_subkv_sweep/run_sweep.sh --budget 16384 --top-ts 1024,4096,8192 --gpu 0
```

> [!TIP]
> **Phase 1 Reuse Optimization:**
> Phase 1 initial compaction ($S=16384$) is executed once during Arm 1 ($t=1024$). Subsequent arms ($t=4096, 8192$) detect and reuse `cache_last.pt` directly, saving substantial compute time.

### Sweeping Over Multiple Datasets

To run the sweep sequentially across multiple datasets (e.g. `qasper`, `quality`, `finqa`, `techqa`):

```bash
for DATASET in qasper quality finqa techqa; do
  echo "======================================================================"
  echo " Starting E2E sweep for: $DATASET"
  echo "======================================================================"
  bash examples/e2e_subkv_sweep/run_sweep.sh --dataset "$DATASET" --gpu 0
done
```

Or fan them out in parallel if multiple GPUs are available:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper  --gpu 0 > qasper_sweep.log 2>&1 &
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset quality --gpu 1 > quality_sweep.log 2>&1 &
wait
```

> [!TIP]
> **Running with an AI Agent:**
> You can prompt the AI agent directly:
> *"Run the sub-KV sweep over datasets `qasper`, `quality`, `finqa` on GPU 0 in the background."*
> The agent will launch the dataset loop as a background task, monitor the execution logs, and notify you when the 5×5 matrices are ready.

### Running in Background (`tmux` / `nohup`)
Because the sweep across 3 budget sizes processes 5 stages per size, it is recommended to run inside `tmux` or with `nohup`:

```bash
nohup bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --gpu 0 > sweep_run.log 2>&1 &
tail -f sweep_run.log
```

### Dry Run (Preview Plan)
To check the planned execution without running computations:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --dry-run
```

---

## 4. Script Options & CLI Arguments

```
Options:
  --dataset <name>     Dataset name: qasper, quality, finqa, techqa (default: qasper)
  --datasets <list>    Comma-separated dataset names to run sequentially
  --model <path>       Base model ID or path (default: Qwen/Qwen3-4B-Instruct-2507)
  --gpu <id>           GPU index for compaction writes (default: 0)
  --eval-gpus <list>   GPU index or comma-separated list for evaluations (default: same as --gpu)
  --budget <size>      Single sub-KV cache budget to sweep across multiple top-ts (e.g. 16384)
  --budgets <list>     Comma-separated sub-KV cache budgets (default: 1024,2048,4096)
  --top-ts <list>      Comma-separated proportional top-t values (default: 64,128,256)
  --max-queries-per-head <N> Upper ceiling on reference queries sampled per head (default: 1024)
  --eval-accuracy      Run additional 5x5 text generation accuracy evaluation (for QuALITY/LongHealth MCQ)
  --force              Rebuild and overwrite existing cache/eval artifacts
  --dry-run            Print execution plan without executing
  -h, --help           Show help message
```

---

## 5. Outputs & Results Inspection

All outputs are structured deterministically:

### Generated Files

1. **Auto-generated Recipes:**
   - Saved under `outputs/recipes/subkv_sweep/e2e_budget{SIZE}_topt{TOP_T}.yaml`.
2. **Phase 1 Initial Compaction Caches:**
   - Saved under `outputs/experiments/subkv_sweep_{DATASET}/e2e_budget{SIZE}_topt{TOP_T}/p01/*/cache_last.pt`.
3. **Continual Stage Runs:**
   - Saved under `outputs/{DATASET}_5phase_runs/` and `outputs/{DATASET}_5phase_state/`.
4. **Execution Logs:**
   - Saved in `logs/e2e_subkv_sweep/{DATASET}_e2e_budget{SIZE}_topt{TOP_T}_{TIMESTAMP}.log`.
5. **5×5 Perplexity Evaluation Matrices:**
   - Location: `outputs/evaluations/{DATASET}/{TAG}/teacher-forced-logppl-v1/`
   - Files:
     - `matrix.json`: Full metrics, backward/forward transfer, and config summary.
     - `matrix.csv`: Tabular 5×5 matrix where rows are stages (p01–p05) and columns are evaluation phases (p1–p5).
6. **5×5 Generation Accuracy & Recorded Generations (with `--eval-accuracy`):**
   - Location: `outputs/evaluations/{DATASET}/{TAG}/accuracy-freeform-mc-options-primeAnswer-v1/`
   - Files:
     - `matrix.csv` / `matrix.json`: Tabular 5×5 MCQ accuracy matrix.
     - `cells/stage-p{i}__eval-p{j}/generations.jsonl`: **Full raw recorded generations for every single question**, containing the prompt, generated answer, reference answer, parsed option, score, correctness, and metadata.

### Inspecting the 5×5 Matrix

To view the generated perplexity matrix for any completed run:

```bash
cat outputs/evaluations/qasper/e2e_budget1024_topt64/teacher-forced-logppl-v1/matrix.csv
```

- **Diagonal ($p_i, p_i$):** Acquisition (performance on phase $i$ immediately after writing phase $i$).
- **Left of Diagonal ($p_j, p_i$ where $j > i$):** Retention (performance on earlier phase $i$ after subsequent phases have been written).
- **Lower values** indicate better perplexity (better language modeling performance).
