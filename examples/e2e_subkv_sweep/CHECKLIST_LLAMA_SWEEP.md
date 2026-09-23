# Llama 3.2 3B Continual Compaction Sweep: LongHealth Accuracy Checklist & Execution Handoff

This document provides a complete checklist, environment setup guide, dataset verification steps, and actionable execution commands for measuring **Multiple-Choice Generation Accuracy** on **LongHealth** across 5 continual stages using **Llama 3.2 3B Instruct** (`meta-llama/Llama-3.2-3B-Instruct`).

---

## 1. Overview & Experimental Configuration

This sweep evaluates continual clinical memory retention and acquisition on **Llama 3.2 3B Instruct** across two sub-KV cache budgets with a fixed **25% update ratio** ($t = S \times 0.25$):

| Arm | Sub-KV Cache Budget ($S$) | Update Budget $\text{top\_t}$ ($t$) | Update Ratio ($t / S$) | Lineage / Tag Identifier |
| :---: | :---: | :---: | :---: | :--- |
| **Arm 1** | **512** | **128** | 25.0% | `llama3_2_3b_budget512_topt128` |
| **Arm 2** | **8192** | **2048** | 25.0% | `llama3_2_3b_budget8192_topt2048` |

### Benchmark Specification: LongHealth (5 Continual Stages)

- **Domain**: Comprehensive synthetic clinical patient records (20 patients total).
- **Task Structure**: 4 patients per stage $\times$ 20 multiple-choice questions = **80 questions per stage** (400 questions total).
- **Stage-to-Patient Mapping**:
  - `p01` (Stage 1): `patient_01`, `patient_02`, `patient_11`, `patient_12`
  - `p02` (Stage 2): `patient_03`, `patient_04`, `patient_13`, `patient_14`
  - `p03` (Stage 3): `patient_05`, `patient_06`, `patient_15`, `patient_16`
  - `p04` (Stage 4): `patient_07`, `patient_08`, `patient_17`, `patient_18`
  - `p05` (Stage 5): `patient_09`, `patient_10`, `patient_19`, `patient_20`
- **Primary Metric**: **Free-form Multiple-Choice Option Generation Accuracy** (5×5 matrix under protocol `accuracy-freeform-mc-options-primeAnswer-v1` using `mc_options` option resolver).
- **Secondary Metric**: Teacher-forced Log-Perplexity (`logppl` 5×5 matrix).

> [!NOTE]
> **Query Ceiling & Regularization Scaling**: All recipes execute with `max_queries_per_head=1024`. To preserve identical relative regularization strength to the baseline of `delta_weight=0.01` with 64 queries, `delta_weight` is explicitly set to `0.16`:
> $$\lambda_\Delta = 0.01 \times \frac{1024}{64} = 0.16$$
> This value is explicitly written into each generated recipe YAML under `objective.delta_weight: 0.16`.

---

## 2. Environment Setup

### Step 1: Hugging Face Authentication (Gated Model Access)
`meta-llama/Llama-3.2-3B-Instruct` is a gated model. Ensure you have accepted the license agreement on Hugging Face and export your access token:

```bash
export HF_TOKEN="hf_your_actual_token_here"
```

To persist it across sessions:
```bash
echo 'export HF_TOKEN="hf_your_actual_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Python Environment & Dependencies
From the repository root (`/home/vo43/gated-continual-cartridges`):

```bash
# 1. Navigate to workspace root
cd /home/vo43/gated-continual-cartridges

# 2. Activate Python environment
conda activate cartridges
# Or virtualenv: source .venv/bin/activate

# 3. Ensure the package is installed in editable mode
pip install -e .

# 4. Verify PyTorch CUDA availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count(), '| Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Step 3: Pre-cache Llama 3.2 3B Weights Offline
The evaluation driver operates under `HF_HUB_OFFLINE=1` for determinism. Pre-download the tokenizer and model weights before launching sweeps:

```bash
python3 -c '
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
print(f"Downloading {model_id}...")
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id)
print("Llama 3.2 3B weights successfully cached locally!")
'
```

---

## 3. Dataset Hydration & Verification

### Step 1: LongHealth Synth & Eval Layout
The 5-stage LongHealth dataset is sourced from:
- Hugging Face repository: `qtris123/gated-continual-cartridges-data` under `data/longhealth/synth/`
- Local path structure:
  - `data/longhealth/synth/p01..p05/self_study-n8192/artifact/dataset.parquet`
  - `data/longhealth/train/qwen_longhealth_p1..p5_task_8192.parquet` (symlinked)
  - `data/longhealth/phases/phase1..5_eval.parquet` (80 questions each, tagged `longhealth_mcq`)
  - `data/longhealth/phases/phase1..5.txt` (full medical record corpus)
  - `data/longhealth/phases/manifest.json`

If regenerating the local phases, run:
```bash
python3 examples/maintenance/data/prepare_longhealth_phases.py
```

### Step 2: Run Pre-Flight Dataset Verification
Verify that all 5 training synth files and 5 evaluation parquets are present:

```bash
python3 -c "
import sys
from examples.shared.am.continual_env import spec

s = spec('longhealth')
missing_eval = [p for p in range(1, 6) if not s.eval_path(p).exists()]
missing_synth = [p for p in range(1, 6) if not s.synth_path(p).exists()]

if missing_eval or missing_synth:
    print('[FAIL] LongHealth dataset incomplete:')
    if missing_eval: print(f'  Missing eval phases: {missing_eval}')
    if missing_synth: print(f'  Missing synth phases: {missing_synth}')
    sys.exit(1)

print('[OK] LongHealth: all 5 eval and 5 synth parquets successfully verified!')
"
```

---

## 4. Execution Checklist (To-Do List)

Track your execution progress:

### Pre-Flight
- [ ] Export `HF_TOKEN` with access to `meta-llama/Llama-3.2-3B-Instruct`.
- [ ] Verify local Python environment (`conda activate cartridges`).
- [ ] Pre-cache Llama 3.2 3B weights locally.
- [ ] Verify LongHealth dataset artifacts (`python3 examples/maintenance/data/prepare_longhealth_phases.py`).
- [ ] Run dry-run verification (`--dry-run`).

### Arm 1: Budget 512, Top-t 128 (25% Update Ratio)
- [ ] Launch Arm 1 compaction & accuracy evaluation on LongHealth:
  ```bash
  bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
    --dataset longhealth \
    --budgets 512 \
    --top-ts 128 \
    --eval-accuracy \
    --gpu 0
  ```
- [ ] Verify Phase 1 cache created: `outputs/experiments/subkv_sweep_longhealth/llama3_2_3b_budget512/p01/*/cache_last.pt`
- [ ] Verify continual stages p02–p05 completed: `outputs/longhealth_5phase_state/llama3_2_3b_budget512_topt128/p05.json`
- [ ] Verify accuracy matrix generated:
  - `outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - `outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/accuracy-freeform-mc-options-primeAnswer-v1/matrix.json`
- [ ] Verify slot frequency artifacts: `outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/slot_frequency/summary.json`

### Arm 2: Budget 8192, Top-t 2048 (25% Update Ratio)
- [ ] Launch Arm 2 compaction & accuracy evaluation on LongHealth:
  ```bash
  bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
    --dataset longhealth \
    --budgets 8192 \
    --top-ts 2048 \
    --eval-accuracy \
    --gpu 0
  ```
- [ ] Verify Phase 1 cache created: `outputs/experiments/subkv_sweep_longhealth/llama3_2_3b_budget8192/p01/*/cache_last.pt`
- [ ] Verify continual stages p02–p05 completed: `outputs/longhealth_5phase_state/llama3_2_3b_budget8192_topt2048/p05.json`
- [ ] Verify accuracy matrix generated:
  - `outputs/evaluations/longhealth/llama3_2_3b_budget8192_topt2048/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - `outputs/evaluations/longhealth/llama3_2_3b_budget8192_topt2048/accuracy-freeform-mc-options-primeAnswer-v1/matrix.json`
- [ ] Verify slot frequency artifacts: `outputs/evaluations/longhealth/llama3_2_3b_budget8192_topt2048/slot_frequency/summary.json`

### Post-Run Analysis
- [ ] Compare 5×5 accuracy matrices between Arm 1 ($S=512$) and Arm 2 ($S=8192$).
- [ ] Inspect forgetting across stages (retention of early patients after training on patients 17–20).
- [ ] Inspect slot overlap geometry and reuse rates in `slot_frequency/jaccard_overlap.csv`.

---

## 5. Execution Commands

### Recommended: Run Both Arms Sequentially (Single Command)

Runs both arms ($S=512, t=128$ and $S=8192, t=2048$) sequentially on LongHealth with automatic generation accuracy evaluation:

```bash
# Description: Execute full 2-arm sweep on LongHealth measuring MCQ accuracy on GPU 0
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0
```

---

### Multi-GPU Evaluation Acceleration

If multiple GPUs are available (e.g. GPU 0 for compaction writes, GPUs 0, 1, 2, 3 for held-out evaluation fan-out):

```bash
# Description: Speed up 5x5 accuracy evaluations by fanning stages across GPUs 0, 1, 2, 3
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0 \
  --eval-gpus 0,1,2,3
```

---

### Dry-Run Verification (Test Before Compute)

Always test that recipe generation, model configuration, and pipeline paths are valid:

```bash
# Description: Dry-run preview for LongHealth accuracy sweep
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --dry-run
```

---

### Running in the Background (`tmux`)

Because running both arms takes several hours, run inside a persistent `tmux` session:

```bash
# 1. Create a persistent session
tmux new -s llama_longhealth

# 2. Inside tmux, export HF_TOKEN and run
export HF_TOKEN="hf_..."
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0

# 3. Detach from session: Press Ctrl+B, then D
# 4. Re-attach anytime:
tmux attach -t llama_longhealth
```

---

## 6. Output Artifacts & Result Inspection

### Result File Locations

For each tag (`{tag}` $\in$ `llama3_2_3b_budget512_topt128`, `llama3_2_3b_budget8192_topt2048`):

1. **LongHealth MCQ Generation Accuracy Matrix** (Primary Result):
   - CSV format: `outputs/evaluations/longhealth/{tag}/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
   - JSON summary: `outputs/evaluations/longhealth/{tag}/accuracy-freeform-mc-options-primeAnswer-v1/matrix.json`
   - Individual question predictions: `outputs/evaluations/longhealth/{tag}/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`

2. **Teacher-Forced Log-Perplexity Matrix**:
   - CSV format: `outputs/evaluations/longhealth/{tag}/teacher-forced-logppl-v1/matrix.csv`
   - JSON summary: `outputs/evaluations/longhealth/{tag}/teacher-forced-logppl-v1/matrix.json`

3. **Slot Frequency & Overlap Geometry Artifacts**:
   - Tensor checkpoint: `outputs/evaluations/longhealth/{tag}/slot_frequency/slot_frequency.pt`
   - Summary metrics JSON: `outputs/evaluations/longhealth/{tag}/slot_frequency/summary.json`
   - Cross-stage Jaccard overlap CSV: `outputs/evaluations/longhealth/{tag}/slot_frequency/jaccard_overlap.csv`
   - Per-layer write distribution: `outputs/evaluations/longhealth/{tag}/slot_frequency/per_layer_stats.csv`

4. **Execution Logs**:
   - Compaction run log: `logs/e2e_subkv_sweep/longhealth_{tag}_*.log`
   - Accuracy evaluation log: `logs/e2e_subkv_sweep/acc_longhealth_{tag}_*.log`

---

### Quick Commands to View Results

#### View LongHealth 5×5 Accuracy Matrix
```bash
# View Arm 1 (Budget 512, Top-t 128) accuracy matrix:
column -s, -t outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# View Arm 2 (Budget 8192, Top-t 2048) accuracy matrix:
column -s, -t outputs/evaluations/longhealth/llama3_2_3b_budget8192_topt2048/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv
```

#### View Cross-Stage Slot Overlap
```bash
# View Jaccard overlap between continual stages:
column -s, -t outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/slot_frequency/jaccard_overlap.csv

# View summary metrics (active slots, reuse rate):
cat outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/slot_frequency/summary.json
```

---

## 7. How to Read the 5×5 Accuracy Matrix

The output 5×5 matrix represents accuracy across the 5 continual stages:

$$\begin{pmatrix}
\text{Acc}(p_1, p_1) & \cdot & \cdot & \cdot & \cdot \\
\text{Acc}(p_2, p_1) & \text{Acc}(p_2, p_2) & \cdot & \cdot & \cdot \\
\text{Acc}(p_3, p_1) & \text{Acc}(p_3, p_2) & \text{Acc}(p_3, p_3) & \cdot & \cdot \\
\text{Acc}(p_4, p_1) & \text{Acc}(p_4, p_2) & \text{Acc}(p_4, p_3) & \text{Acc}(p_4, p_4) & \cdot \\
\text{Acc}(p_5, p_1) & \text{Acc}(p_5, p_2) & \text{Acc}(p_5, p_3) & \text{Acc}(p_5, p_4) & \text{Acc}(p_5, p_5)
\end{pmatrix}$$

- **Diagonal $(p_i, p_i)$ [Acquisition]**: Multiple-choice accuracy on phase $i$'s patients immediately after being compacted in stage $i$.
- **Lower Triangle $(p_j, p_i)$ with $j > i$ [Retention / Forgetting]**: Multiple-choice accuracy on earlier phase $i$'s patients after subsequent patient cohorts ($i+1 \dots j$) have been incrementally compacted into the sub-KV cache.
