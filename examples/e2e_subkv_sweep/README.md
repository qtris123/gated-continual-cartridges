# Continual AM Overlap Sweep (Encouraging Slot Reuse Across Stages)

This directory contains the automated pipeline script and runbook for sweeping over the **overlap bonus parameter ($\lambda$)** across 5 continual stages on **QuALITY** and **QASPER**.

---

## Overview & Hypothesis

In continual attention matching (AM), each new document or stage selects a subset of slots ($\text{top\_t}$) to update. By default or under soft-locality penalties ($-\lambda$), previously written slots are down-weighted to enforce slot isolation.

Here, we **reverse the exponent** in `SlotSelector._apply_usage_penalty` in [`cartridges/am/components/slots.py`](../../cartridges/am/components/slots.py) to **encourage slot reuse and cross-stage overlap**:

$$\text{access\_score}' = \text{access\_score} \times (1 + \text{usage})^\lambda$$

- **$\lambda = 0.0$**: Baseline (no overlap bonus, bit-identical to standard continual AM).
- **$\lambda > 0.0$**: Progressively boosts the priority of previously written slots, encouraging successive stages to write into existing cartridge slots rather than colonizing unused slots.

### Sweep Matrix ($\lambda$ Arms)

We sweep $\lambda \in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]$ over all 5 continual stages for both **QuALITY** and **QASPER**:

| Arm | $\lambda$ | Tag / Label | Behavior | Status |
| :---: | :---: | :--- | :--- | :--- |
| **1** | **0.0** | `overlap_lam0` | Baseline (standard selection, no overlap bonus) | Ready |
| **2** | **0.25** | `overlap_lam0p25` | Mild overlap incentive | Ready |
| **3** | **0.5** | `overlap_lam0p5` | Moderate overlap incentive | Ready |
| **4** | **1.0** | `overlap_lam1` | Proportional overlap incentive | Ready |
| **5** | **2.0** | `overlap_lam2` | Strong overlap incentive | Ready |
| **6** | **4.0** | `overlap_lam4` | Aggressive overlap incentive | Ready |

- **Sub-KV Cache Budget:** $S = 1024$ slots (Phase 1 initial compaction built once per dataset and reused across all $\lambda$ arms).
- **Update Budget ($\text{top\_t}$):** $t = 64$ slots per layer per document (6.25% proportional write ratio).
- **Evaluation:** Full 5×5 stage-by-eval perplexity matrix (`matrix.csv`) after every stage.

---

## 1. Environment Setup

### System Requirements
- Linux OS (Ubuntu 20.04/22.04 or RHEL/CentOS/Rocky 8/9).
- NVIDIA GPU with at least 24 GB VRAM (RTX 3090/4090, A5000, A100, H100, etc.).
- Python 3.10, 3.11, or 3.12 with CUDA drivers.
- `zstd` utility installed (`sudo apt-get install zstd` or `sudo dnf install zstd`).

### Step 1: Install Dependencies

```bash
# From the repository root (gated-continual-cartridges):
pip install --upgrade pip
pip install -e .
```

### Step 2: Download Benchmark Datasets

If dataset files are not yet present in `data/`, download them from Hugging Face:

```bash
export HF_TOKEN="hf_your_token_here"
bash scripts/prepare_artifacts.sh --which data
```

Verify that `quality` and `qasper` are ready:

```bash
python3 -c '
from examples.shared.am.continual_env import spec
for ds in ["quality", "qasper"]:
    s = spec(ds)
    ok = all(s.synth_path(p).exists() and s.eval_path(p).exists() for p in range(1, 6))
    print(f"[{chr(10003) if ok else chr(10007)}] {ds}")
'
```

---

## 2. Running the Overlap Sweep

The automated runner script is located at [`examples/e2e_subkv_sweep/run_sweep.sh`](run_sweep.sh).

### Quick Start (Run Both Datasets on GPU 0)

To sweep over all $\lambda \in [0, 0.25, 0.5, 1.0, 2.0, 4.0]$ across 5 stages for both **QuALITY** and **QASPER**:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh --gpu 0
```

### Running on a Single Dataset

```bash
# QuALITY only:
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset quality --gpu 0

# QASPER only:
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper --gpu 0
```

### Running in Parallel (Multi-GPU Setup)

If you have 2 GPUs, run QuALITY and QASPER simultaneously:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset quality --gpu 0 > quality_overlap.log 2>&1 &
bash examples/e2e_subkv_sweep/run_sweep.sh --dataset qasper  --gpu 1 > qasper_overlap.log 2>&1 &
wait
```

### Running in the Background (`nohup` / `tmux`)

```bash
nohup bash examples/e2e_subkv_sweep/run_sweep.sh --gpu 0 > overlap_sweep.log 2>&1 &
tail -f overlap_sweep.log
```

### Dry Run (Preview Plan)

To verify the execution plan, recipes, and paths without executing computations:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh --dry-run
```

---

## 3. Alternative: Running via Sweep Manifest (`sweep.py`)

You can also use the repository's native manifest launcher:

```bash
M=examples/shared/am/manifests/overlap_sweep.yaml

# 1. Preview resolved arms:
python -m examples.shared.am.sweep arms --manifest $M --stream lambda

# 2. Render recipe YAMLs:
python -m examples.shared.am.sweep render --manifest $M --stream lambda

# 3. Launch on Quality:
python -m examples.shared.am.sweep launch --manifest $M --stream lambda --dataset quality --gpus 0

# 4. Launch on QASPER:
python -m examples.shared.am.sweep launch --manifest $M --stream lambda --dataset qasper --gpus 0
```

---

## 4. Script Options & CLI Arguments

```
Usage: examples/e2e_subkv_sweep/run_sweep.sh [options]

Options:
  --datasets <list>    Comma-separated datasets: quality, qasper (default: quality,qasper)
  --dataset <name>     Single dataset alias: quality or qasper
  --lambdas <list>     Comma-separated lambda values for overlap bonus (default: 0.0,0.25,0.5,1.0,2.0,4.0)
  --budget <size>      Sub-KV cache budget for Phase 1 (default: 1024)
  --top-t <int>        Proportional top-t updates per layer (default: 64)
  --gpu <id>           GPU index for compaction writes (default: 0)
  --eval-gpus <list>   GPU index or comma-separated list for evaluations (default: same as --gpu)
  --force              Rebuild and overwrite existing cache/eval artifacts
  --dry-run            Print execution plan without running
  -h, --help           Show help message
```

---

## 5. Inspecting Outputs & Results

All outputs are saved deterministically:

### 1. Generated Recipes
- Location: `outputs/recipes/overlap_sweep/`
- Files:
  - `p01_budget1024.yaml`: Initial compaction recipe.
  - `overlap_lam{LAM}.yaml`: Continual AM recipe with `usage_penalty_lambda: {LAM}`.

### 2. Execution Logs
- Location: `logs/overlap_sweep/`
- Pattern: `{dataset}_overlap_lam{LAM}_{timestamp}.log`

### 3. 5×5 Evaluation Matrices
- Location: `outputs/evaluations/{dataset}/overlap_lam{LAM}/teacher-forced-logppl-v1/`
- Files:
  - `matrix.csv`: Tabular 5×5 matrix where rows are stages (p01–p05) and columns are evaluation phases (p1–p5).
  - `matrix.json`: Detailed metrics including backward transfer (BWT), forward transfer (FWT), and perplexity values.

### Viewing Results

To inspect the 5×5 perplexity matrix for any completed run:

```bash
# Example: Quality with lambda=0.5
cat outputs/evaluations/quality/overlap_lam0p5/teacher-forced-logppl-v1/matrix.csv

# Example: QASPER baseline (lambda=0)
cat outputs/evaluations/qasper/overlap_lam0/teacher-forced-logppl-v1/matrix.csv
```

- **Diagonal ($p_i, p_i$):** Acquisition performance on phase $i$ immediately after writing phase $i$.
- **Lower Triangle ($p_j, p_i$ where $j > i$):** Retention on phase $i$ after subsequent phases have been written.
- **Comparison:** Compare $j > i$ cells across $\lambda = 0.0 \to 4.0$ to see whether encouraging slot overlap improves retention against catastrophic forgetting.
