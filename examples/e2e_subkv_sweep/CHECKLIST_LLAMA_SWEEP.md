# Llama 3.2 3B Continual Compaction Sweep: Checklist & Execution Handoff

This document provides a complete checklist, environment setup guide, dataset hydration steps, and actionable execution commands for running the **End-to-End Sub-KV Cache Compaction Sweep** on **Llama 3.2 3B Instruct** (`meta-llama/Llama-3.2-3B-Instruct`) across **QASPER**, **TechQA**, **FinQA**, **QuALITY**, and **LongHealth** (measuring MCQ generation accuracy).

---

## 1. Overview & Experimental Configuration

This sweep evaluates continual memory retention and acquisition on **Llama 3.2 3B Instruct** across two specific sub-KV cache budgets with a fixed **25% update ratio** ($t = S \times 0.25$):

| Arm | Sub-KV Cache Budget ($S$) | Update Budget $\text{top\_t}$ ($t$) | Update Ratio ($t / S$) | Lineage / Tag Identifier |
| :---: | :---: | :---: | :---: | :--- |
| **Arm 1** | **512** | **128** | 25.0% | `llama3_2_3b_budget512_topt128` |
| **Arm 2** | **8192** | **2048** | 25.0% | `llama3_2_3b_budget8192_topt2048` |

### Target Benchmarks & Evaluation Modes

| Benchmark | Domain / Task Structure | Primary Evaluation Mode | Runner Flag |
| :--- | :--- | :--- | :--- |
| **QASPER** (`qasper`) | 5 NLP tasks (QA, MT, SA, ASR, KG) | Teacher-forced Log-Perplexity (`logppl` 5×5 matrix) | Default |
| **TechQA** (`techqa`) | Technical support technotes (5 cohorts) | Teacher-forced Log-Perplexity (`logppl` 5×5 matrix) | Default |
| **FinQA** (`finqa`) | Financial reports & tables (5 cohorts) | Teacher-forced Log-Perplexity (`logppl` 5×5 matrix) | Default |
| **QuALITY** (`quality`) | Long-document story QA (5 phases) | **Multiple-Choice Option Generation Accuracy** (`mcq` 5×5 matrix) + Log-Perplexity | `--eval-accuracy` (auto-enabled) |
| **LongHealth** (`longhealth`) | Clinical patient records (5 cohorts, patients 01–20) | **Multiple-Choice Option Generation Accuracy** (`mcq` 5×5 matrix) + Log-Perplexity | `--eval-accuracy` (auto-enabled) |

### LongHealth 5-Stage Patient Mapping (Sequential)
- `p01` (Stage 1): `patient_01`, `patient_02`, `patient_03`, `patient_04` (80 questions)
- `p02` (Stage 2): `patient_05`, `patient_06`, `patient_07`, `patient_08` (80 questions)
- `p03` (Stage 3): `patient_09`, `patient_10`, `patient_11`, `patient_12` (80 questions)
- `p04` (Stage 4): `patient_13`, `patient_14`, `patient_15`, `patient_16` (80 questions)
- `p05` (Stage 5): `patient_17`, `patient_18`, `patient_19`, `patient_20` (80 questions)

> [!NOTE]
> **Query Ceiling & Regularization Scaling**: All recipes and runner commands execute with `max_queries_per_head=1024`. To preserve identical relative regularization strength to the original baseline of `delta_weight=0.01` with 64 queries, `delta_weight` is explicitly set to `0.16` in `run_sweep.sh`:
> $$\lambda_\Delta = 0.01 \times \frac{1024}{64} = 0.16$$
> This value is explicitly written into and logged in each experiment's recipe YAML under `objective.delta_weight: 0.16`.

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

### Step 1: Hydrate Dataset Artifacts
The continual AM pipeline requires:
1. Per-phase evaluation parquets (`data/{dataset}/phases/phase{1..5}_eval.parquet`)
2. In-context self-study reference query parquets (`data/{dataset}/train/qwen_*_task_8192.parquet` or `data/{dataset}/synth/`)

Hydrate missing dataset shards from Hugging Face:
```bash
bash scripts/prepare_artifacts.sh --which data
```

### Step 2: LongHealth Preparation
Ensure the 5-phase LongHealth eval parquets and train symlinks are materialized:
```bash
python3 examples/maintenance/data/prepare_longhealth_phases.py
```

### Step 3: Run Pre-Flight Dataset Verification
Execute this check script to verify that all 5 eval phases and 5 training synth files exist for all 5 target datasets:

```bash
python3 -c "
import sys
from examples.shared.am.continual_env import spec

datasets = ['qasper', 'techqa', 'finqa', 'quality', 'longhealth']
all_ok = True

for ds in datasets:
    s = spec(ds)
    missing_eval = [p for p in range(1, 6) if not s.eval_path(p).exists()]
    missing_synth = [p for p in range(1, 6) if not s.synth_path(p).exists()]
    if missing_eval or missing_synth:
        all_ok = False
        print(f'[FAIL] {ds}:')
        if missing_eval: print(f'  Missing eval phases: {missing_eval}')
        if missing_synth: print(f'  Missing synth phases: {missing_synth}')
    else:
        print(f'[OK] {ds}: all 5 eval and 5 synth parquets found.')

if not all_ok:
    print('\nAction required: Run `bash scripts/prepare_artifacts.sh --which data`')
    sys.exit(1)
print('\nAll 5 datasets ready for sweep execution!')
"
```

---

## 4. Execution Checklist (To-Do List)

Track your execution progress:

### Pre-Flight
- [ ] Export `HF_TOKEN` with access to `meta-llama/Llama-3.2-3B-Instruct`.
- [ ] Verify local Python environment and editable install (`pip install -e .`).
- [ ] Pre-cache Llama 3.2 3B weights locally.
- [ ] Hydrate dataset artifacts (`scripts/prepare_artifacts.sh --which data`).
- [ ] Prepare LongHealth phases (`python3 examples/maintenance/data/prepare_longhealth_phases.py`).
- [ ] Run pre-flight dataset verification script (all 5 datasets `[OK]`).
- [ ] Run `--dry-run` to verify recipe generation and execution plans.

### Log-Perplexity Sweeps (`qasper`, `techqa`, `finqa`)
- [ ] Launch QASPER sweep (`budget 512, top 128` & `budget 8192, top 2048`).
  - [ ] Verify `outputs/evaluations/qasper/llama3_2_3b_budget512_topt128/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify `outputs/evaluations/qasper/llama3_2_3b_budget8192_topt2048/teacher-forced-logppl-v1/matrix.csv`
- [ ] Launch TechQA sweep (`budget 512, top 128` & `budget 8192, top 2048`).
  - [ ] Verify `outputs/evaluations/techqa/llama3_2_3b_budget512_topt128/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify `outputs/evaluations/techqa/llama3_2_3b_budget8192_topt2048/teacher-forced-logppl-v1/matrix.csv`
- [ ] Launch FinQA sweep (`budget 512, top 128` & `budget 8192, top 2048`).
  - [ ] Verify `outputs/evaluations/finqa/llama3_2_3b_budget512_topt128/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify `outputs/evaluations/finqa/llama3_2_3b_budget8192_topt2048/teacher-forced-logppl-v1/matrix.csv`

### Multiple-Choice Option Generation Accuracy Sweeps (`quality`, `longhealth`)
- [ ] Launch QuALITY sweep with MCQ generation accuracy evaluation (`budget 512, top 128` & `budget 8192, top 2048`).
  - [ ] Verify logppl: `outputs/evaluations/quality/llama3_2_3b_budget512_topt128/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify logppl: `outputs/evaluations/quality/llama3_2_3b_budget8192_topt2048/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/quality/llama3_2_3b_budget512_topt128/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/quality/llama3_2_3b_budget8192_topt2048/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify individual predictions: `outputs/evaluations/quality/*/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
- [ ] Launch LongHealth sweep with MCQ generation accuracy evaluation (`budget 512, top 128` & `budget 8192, top 2048`).
  - [ ] Verify logppl: `outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify logppl: `outputs/evaluations/longhealth/llama3_2_3b_budget8192_topt2048/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/longhealth/llama3_2_3b_budget8192_topt2048/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify individual predictions: `outputs/evaluations/longhealth/*/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`

### Post-Run & Archival
- [ ] Inspect 5×5 matrices (check diagonal acquisition and off-diagonal retention).
- [ ] Inspect slot frequency and overlap geometry (`outputs/evaluations/{dataset}/{tag}/slot_frequency/summary.json`).
- [ ] Commit or archive final evaluation matrices in `outputs/evaluations/`.

---

## 5. Execution Commands (25% Update Ratio)

### Strategy 1: Recommended Batch Execution (2 Sequential Runs)

#### Batch 1: Run Logppl Benchmarks (`qasper`, `techqa`, `finqa`)
Runs both arms ($S=512, t=128$ and $S=8192, t=2048$) sequentially across QASPER, TechQA, and FinQA on GPU 0:

```bash
# Description: Execute 25% update ratio sweeps across qasper, techqa, finqa sequentially on GPU 0
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --datasets qasper,techqa,finqa \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --gpu 0
```

#### Batch 2: Run Multiple-Choice Accuracy Benchmarks (`quality`, `longhealth`)
Runs both arms on QuALITY and LongHealth, executing the 5×5 teacher-forced logppl evaluation followed by the 5×5 free-form MCQ generation accuracy matrix evaluation:

```bash
# Description: Execute QuALITY and LongHealth sweeps (25% ratio) with held-out text generation accuracy matrix on GPU 0
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --datasets quality,longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0
```

---

### Strategy 2: Individual Per-Dataset Commands

If running on separate GPUs or monitoring each dataset independently:

#### 1. QASPER (Logppl)
> **Description**: Runs Arm 1 ($S=512, t=128$) and Arm 2 ($S=8192, t=2048$) on QASPER NLP tasks.
```bash
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset qasper \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --gpu 0
```

#### 2. TechQA (Logppl)
> **Description**: Runs both arms on technical documentation cohorts.
```bash
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset techqa \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --gpu 0
```

#### 3. FinQA (Logppl)
> **Description**: Runs both arms on financial earnings report cohorts.
```bash
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset finqa \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --gpu 0
```

#### 4. QuALITY (Logppl + Generation Accuracy)
> **Description**: Runs both arms on QuALITY long-form stories and scores multiple-choice questions across all 5 continual stages.
```bash
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset quality \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0
```

#### 5. LongHealth (Logppl + Generation Accuracy)
> **Description**: Runs both arms on LongHealth clinical records and scores multiple-choice questions across all 5 continual stages.
```bash
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --dataset longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0
```

---

### Strategy 3: Multi-GPU Evaluation Fan-Out (Faster Runs)

If multiple GPUs are available (e.g. GPU 0 for compaction writes, GPUs 0, 1, 2, 3 for evaluations):

```bash
# Description: Accelerate held-out 5x5 accuracy evaluations for QuALITY and LongHealth across GPUs 0, 1, 2, 3
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --datasets quality,longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0 \
  --eval-gpus 0,1,2,3
```

---

### Strategy 4: Dry-Run Verification (Test Before Compute)

Always test that recipe generation, paths, and arguments are valid without launching heavy GPU jobs:

```bash
# Description: Dry-run preview for logppl benchmarks (25% update ratio)
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --datasets qasper,techqa,finqa \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --dry-run

# Description: Dry-run preview for QuALITY and LongHealth with accuracy (25% update ratio)
bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --datasets quality,longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --dry-run
```

---

### Strategy 5: Running in the Background (`tmux` / `nohup`)

Because full sweeps take several hours, launch them in persistent background sessions:

#### Using `tmux` (Recommended)
```bash
# 1. Create a persistent session
tmux new -s llama_mcq_sweep

# 2. Inside tmux, export HF_TOKEN and run
export HF_TOKEN="hf_..."
bash examples/e2e_subkv_sweep/run_sweep_llama.sh --datasets quality,longhealth --budgets 512,8192 --top-ts 128,2048 --eval-accuracy --gpu 0

# 3. Detach from session: Press Ctrl+B, then D
# 4. Re-attach anytime:
tmux attach -t llama_mcq_sweep
```

#### Using `nohup`
```bash
nohup bash examples/e2e_subkv_sweep/run_sweep_llama.sh \
  --datasets quality,longhealth \
  --budgets 512,8192 \
  --top-ts 128,2048 \
  --eval-accuracy \
  --gpu 0 > logs_mcq_sweep.out 2>&1 &

# Monitor execution:
tail -f logs_mcq_sweep.out
```

---

## 6. Output Artifacts & Result Inspection

### Result File Locations

For any dataset (`{dataset}` $\in$ `qasper`, `techqa`, `finqa`, `quality`, `longhealth`):

1. **Teacher-Forced Log-Perplexity Matrix**:
   - CSV format: `outputs/evaluations/{dataset}/{tag}/teacher-forced-logppl-v1/matrix.csv`
   - JSON summary: `outputs/evaluations/{dataset}/{tag}/teacher-forced-logppl-v1/matrix.json`

2. **Multiple-Choice Option Generation Accuracy Matrix** (`quality`, `longhealth`):
   - CSV format: `outputs/evaluations/{dataset}/{tag}/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
   - JSON summary: `outputs/evaluations/{dataset}/{tag}/accuracy-freeform-mc-options-primeAnswer-v1/matrix.json`
   - Individual generated predictions: `outputs/evaluations/{dataset}/{tag}/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`

3. **Slot Frequency & Overlap Geometry Artifacts**:
   - PyTorch tensor archive: `outputs/evaluations/{dataset}/{tag}/slot_frequency/slot_frequency.pt`
   - Summary metrics JSON: `outputs/evaluations/{dataset}/{tag}/slot_frequency/summary.json`
   - Stage-by-stage Jaccard overlap CSV: `outputs/evaluations/{dataset}/{tag}/slot_frequency/jaccard_overlap.csv`
   - Layer-by-layer write statistics CSV: `outputs/evaluations/{dataset}/{tag}/slot_frequency/per_layer_stats.csv`

4. **Execution Logs**:
   - Master sweep log: `logs/e2e_subkv_sweep/{dataset}_{tag}_*.log`
   - Accuracy generation log: `logs/e2e_subkv_sweep/acc_{dataset}_{tag}_*.log`

5. **Auto-Generated Recipes**:
   - Arm 1: `outputs/recipes/subkv_sweep/llama3_2_3b_budget512_topt128.yaml`
   - Arm 2: `outputs/recipes/subkv_sweep/llama3_2_3b_budget8192_topt2048.yaml`

---

### Quick Commands to View Results

#### View Log-Perplexity 5×5 Matrix
```bash
# View Arm 1 (512/128) on QASPER:
column -s, -t outputs/evaluations/qasper/llama3_2_3b_budget512_topt128/teacher-forced-logppl-v1/matrix.csv

# View Arm 2 (8192/2048) on QASPER:
column -s, -t outputs/evaluations/qasper/llama3_2_3b_budget8192_topt2048/teacher-forced-logppl-v1/matrix.csv
```

#### View MCQ Accuracy 5×5 Matrix (`quality` & `longhealth`)
```bash
# View Arm 1 (512/128) QuALITY accuracy:
column -s, -t outputs/evaluations/quality/llama3_2_3b_budget512_topt128/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# View Arm 2 (8192/2048) QuALITY accuracy:
column -s, -t outputs/evaluations/quality/llama3_2_3b_budget8192_topt2048/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# View Arm 1 (512/128) LongHealth accuracy:
column -s, -t outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# View Arm 2 (8192/2048) LongHealth accuracy:
column -s, -t outputs/evaluations/longhealth/llama3_2_3b_budget8192_topt2048/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv
```

#### View Slot Frequency & Cross-Stage Overlap
```bash
# View cross-stage Jaccard overlap matrix:
column -s, -t outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/slot_frequency/jaccard_overlap.csv

# View layer-by-layer slot write counts and coverage:
column -s, -t outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/slot_frequency/per_layer_stats.csv

# View JSON summary metrics:
cat outputs/evaluations/longhealth/llama3_2_3b_budget512_topt128/slot_frequency/summary.json
```

---

## 7. How to Read the 5×5 Evaluation Matrix

The output 5×5 matrix represents performance across the 5 continual stages:

$$\begin{pmatrix}
(p_1, p_1) & \cdot & \cdot & \cdot & \cdot \\
(p_2, p_1) & (p_2, p_2) & \cdot & \cdot & \cdot \\
(p_3, p_1) & (p_3, p_2) & (p_3, p_3) & \cdot & \cdot \\
(p_4, p_1) & (p_4, p_2) & (p_4, p_3) & (p_4, p_4) & \cdot \\
(p_5, p_1) & (p_5, p_2) & (p_5, p_3) & (p_5, p_4) & (p_5, p_5)
\end{pmatrix}$$

- **Diagonal $(p_i, p_i)$ [Acquisition]**: Loss / Accuracy on task $i$ immediately after being compacted/written in stage $i$.
- **Lower Triangle $(p_j, p_i)$ with $j > i$ [Retention / Forgetting]**: Performance on early task $i$ after subsequent tasks ($i+1 \dots j$) have been incrementally compacted into the sub-KV cache. Minimal degradation indicates robust continual memory.
