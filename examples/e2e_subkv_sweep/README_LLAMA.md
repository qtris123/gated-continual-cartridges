# End-to-End Continual Compaction Sweep on Llama 3.2 3B

This guide provides instructions for running the **End-to-End Sub-KV Cache Compaction Sweep** using **Llama 3.2 3B Instruct** (`meta-llama/Llama-3.2-3B-Instruct`), replicating the sweep procedure previously conducted on **Qwen 3 4B Instruct**.

---

## 1. Background & Comparison with Qwen

In the Llama 3.2 3B sweep, we evaluate continual memory compaction across **sub-KV cache budgets** ($S = 512, 1024, 2048, 4096$) with a proportional **6.25% update budget** ($t = 32, 64, 128, 256$):

| Budget ($S$) | Update Budget $\text{top\_t}$ ($t$) | Ratio ($t / S$) | Stage 1 (Initial) | Stages 2–5 (Continual AM) |
| :---: | :---: | :---: | :--- | :--- |
| **512** | **32** | 6.25% | Closed-form Arm-D ($N=512$) | Continual 5-stage chain ($t=32$) |
| **1024** | **64** | 6.25% | Closed-form Arm-D ($N=1024$) | Continual 5-stage chain ($t=64$) |
| **2048** | **128** | 6.25% | Closed-form Arm-D ($N=2048$) | Continual 5-stage chain ($t=128$) |
| **4096** | **256** | 6.25% | Closed-form Arm-D ($N=4096$) | Continual 5-stage chain ($t=256$) |

### Existing Qwen Baseline Reference (QASPER)

For comparison, here are the resulting teacher-forced log-perplexity 5×5 matrices achieved on Qwen 3 4B on QASPER:

- **Reference Arm: $S=512$, $t=512$ (Full KV baseline)**:
  ```
  stage, qa,     mt,     sa,     asr,    kg
  p01,   4.0740, 4.0950, 4.1812, 3.7362, 3.9427
  p02,   4.0361, 3.9273, 4.0345, 3.7615, 3.8551
  p03,   3.5347, 3.4276, 3.5231, 3.2999, 3.3887
  p04,   3.6375, 3.5183, 3.6158, 3.3840, 3.5047
  p05,   3.5330, 3.5201, 3.5736, 3.3573, 3.4394
  ```

- **Arm 1: $S=1024$, $t=64$ (`topt64_v1`)**:
  ```
  stage, qa,     mt,     sa,     asr,    kg
  p01,   4.0740, 4.0950, 4.1812, 3.7362, 3.9427
  p02,   4.0528, 3.8803, 4.0638, 3.7659, 3.8540
  p03,   3.4594, 3.3370, 3.4915, 3.1813, 3.2745
  p04,   3.7078, 3.5448, 3.7065, 3.4150, 3.5222
  p05,   3.5602, 3.4847, 3.6151, 3.3440, 3.3904
  ```

- **Arm 2: $S=2048$, $t=128$ (`topt128_v1`)**:
  ```
  stage, qa,     mt,     sa,     asr,    kg
  p01,   4.0740, 4.0950, 4.1812, 3.7362, 3.9427
  p02,   3.8054, 3.6766, 3.8188, 3.5443, 3.6524
  p03,   3.4165, 3.3164, 3.4255, 3.1742, 3.2900
  p04,   3.5789, 3.4994, 3.6051, 3.3316, 3.4929
  p05,   3.6213, 3.6166, 3.6929, 3.4369, 3.5194
  ```

The goal of this Llama run is to test whether the same continual compaction mechanics exhibit comparable or superior retention and acquisition dynamics on Llama 3.2 3B.

---

## 2. What Changes for Llama 3.2 3B?

1. **Model Architecture**:
   - Uses `meta-llama/Llama-3.2-3B-Instruct` wrapped via `FlexLlamaForCausalLM`.
   - 28 layers, 24 query heads, 8 KV heads (Grouped Query Attention).
2. **RoPE Base Frequency**:
   - Qwen uses `rope_theta = 5000000.0` ($5\times 10^6$).
   - Llama 3.2 3B uses `rope_theta = 500000.0` ($5\times 10^5$).
   - Handled automatically via `rope_theta: model` in the recipe.
3. **Reference Queries**:
   - Per experimental design, we reuse the existing self-study reference queries (`data/{dataset}/train/qwen_*_8192.parquet`).
4. **Artifact Namespacing**:
   - Tags are prefixed with `llama3_2_3b_` (e.g. `llama3_2_3b_budget1024_topt64`), ensuring existing Qwen results are untouched.

---

## 3. Environment & Prerequisites

### Step 1: Hugging Face Access (Gated Model)
Because `meta-llama/Llama-3.2-3B-Instruct` is gated, accept the license on Hugging Face and export your token:
```bash
export HF_TOKEN="hf_your_huggingface_token"
```

### Step 2: Python Environment Setup
From the repository root (`gated-continual-cartridges`):
```bash
# Activate existing environment or create a new one
source .venv/bin/activate

# Ensure package is installed in editable mode
pip install -e .
```

### Step 3: Hydrate Dataset Artifacts
If not already downloaded, fetch the evaluation sets and self-study parquets:
```bash
bash scripts/prepare_artifacts.sh --which data
```

### Step 4: Pre-cache Llama 3.2 3B Weights
Because the evaluation driver operates with `HF_HUB_OFFLINE=1` for determinism, pre-download the model weights once:
```bash
python3 -c '
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
print(f"Downloading {model_id}...")
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id)
print("Model ready!")
'
```

---

## 4. Running the Sweep

### Option A: Using the Dedicated Llama Runner (`run_sweep_llama.sh`)

A dedicated convenience script [`run_sweep_llama.sh`](run_sweep_llama.sh) is provided with Llama defaults:

```bash
# Run on GPU 0 for QASPER:
bash examples/e2e_subkv_sweep/run_sweep_llama.sh --dataset qasper --gpu 0

# Run on Quality:
bash examples/e2e_subkv_sweep/run_sweep_llama.sh --dataset quality --gpu 0
```

### Option B: Using the General Runner (`run_sweep.sh`)

You can also invoke `run_sweep.sh` explicitly specifying `--model`:

```bash
bash examples/e2e_subkv_sweep/run_sweep.sh \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset qasper \
  --gpu 0
```

### Option C: Multi-GPU Evaluation Fan-Out

To accelerate evaluation by fanning out the 5 per-stage held-out evaluations across multiple GPUs:

```bash
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset qasper \
  --gpu 0 \
  --eval-gpus 0,1
```

### Running in the Background (`nohup` / `tmux`)

The full 4-arm sweep takes a few hours to complete. Run inside `tmux` or using `nohup`:

```bash
nohup bash examples/e2e_subkv_sweep/run_sweep_llama.sh --dataset qasper --gpu 0 > llama_sweep_qasper.log 2>&1 &
tail -f llama_sweep_qasper.log
```

### Dry Run (Preview Plan)
To inspect the plan and generated recipes without executing GPU computations:
```bash
bash examples/e2e_subkv_sweep/run_sweep_llama.sh --dataset qasper --dry-run
```

---

## 5. Outputs & Result Inspection

### Where Results Live

1. **Perplexity 5×5 Matrix**:
   `outputs/evaluations/{dataset}/llama3_2_3b_budget{SIZE}_topt{TOP_T}/teacher-forced-logppl-v1/matrix.csv`
2. **Full JSON Summary & Metadata**:
   `outputs/evaluations/{dataset}/llama3_2_3b_budget{SIZE}_topt{TOP_T}/teacher-forced-logppl-v1/matrix.json`
3. **Execution Logs**:
   `logs/e2e_subkv_sweep/{dataset}_llama3_2_3b_budget{SIZE}_topt{TOP_T}_*.log`
4. **Auto-generated Recipes**:
   `outputs/recipes/subkv_sweep/llama3_2_3b_budget{SIZE}_topt{TOP_T}.yaml`

### Viewing Results

To display the 5×5 matrix for budget 1024 ($t=64$):
```bash
cat outputs/evaluations/qasper/llama3_2_3b_budget1024_topt64/teacher-forced-logppl-v1/matrix.csv
```

- **Diagonal ($p_i, p_i$)**: Acquisition loss on stage $i$ immediately after write.
- **Left of Diagonal ($p_j, p_i$ where $j > i$)**: Retention loss on stage $i$ after subsequent phases are written.
- Compare these numbers against the Qwen baseline tables above to evaluate cross-architecture performance!
