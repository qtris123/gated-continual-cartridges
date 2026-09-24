# Qwen 3 4B LongHealth Continual Compaction: Ablation Checklist & Execution Handoff

This document provides a complete checklist, environment setup guide, dataset hydration steps, and actionable execution commands for running the **4-Arm Ablation Study** on **Qwen 3 4B Instruct** (`Qwen/Qwen3-4B-Instruct-2507`) on **LongHealth** across 5 sequential patient stages, evaluating both **Multiple-Choice Option Generation Accuracy** (`mcq` 5×5 matrix) and **Teacher-Forced Log-Perplexity** (`logppl` 5×5 matrix).

---

## 1. Overview & Experimental Configuration

This ablation evaluates continual memory retention and acquisition on **Qwen 3 4B** for the LongHealth benchmark (the missing dataset in the Qwen suite), matching the naming conventions of the Llama sweeps (`qwen3_4b_budget...` $\leftrightarrow$ `llama3_2_3b_budget...`).

All 4 runs use a fixed sub-KV cache budget $S = 512$, sweeping over update budgets $\text{top\_t} \in \{32, 64\}$ across two query ceiling and delta regularization regimes:

1. **Group A (Baseline Regime)**: `max_queries_per_head = 64`, `delta_weight = 0.01` ($\lambda_\Delta = 0.01$).
2. **Group B (Scaled Regime)**: `max_queries_per_head = 1024`, `delta_weight = 0.16` ($\lambda_\Delta = 0.01 \times \frac{1024}{64} = 0.16$).

### 4-Arm Matrix Summary

| Arm | Sub-KV Budget ($S$) | Update Budget $\text{top\_t}$ ($t$) | Update Ratio ($t / S$) | Max Queries ($N_q$) | Regularization ($\lambda_\Delta$) | Tag Suffix | Lineage / Tag Identifier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arm 1 (A1)** | **512** | **32** | 6.25% | **64** | **0.01** | `_q64` | `qwen3_4b_budget512_topt32_q64` |
| **Arm 2 (A2)** | **512** | **64** | 12.50% | **64** | **0.01** | `_q64` | `qwen3_4b_budget512_topt64_q64` |
| **Arm 3 (B1)** | **512** | **32** | 6.25% | **1024** | **0.16** | `_q1024` | `qwen3_4b_budget512_topt32_q1024` |
| **Arm 4 (B2)** | **512** | **64** | 12.50% | **1024** | **0.16** | `_q1024` | `qwen3_4b_budget512_topt64_q1024` |

### Key Scientific Questions Addressed by this Ablation
1. **Query Ceiling & Regularization**: Does scaling reference queries from 64 to 1024 with proportional regularization ($\lambda_\Delta = 0.16$) improve acquisition while preserving clinical record retention across stages?
2. **Update Budget Sensitivity**: Does doubling the update ratio from 6.25% ($t=32$) to 12.5% ($t=64$) reduce forgetting of early patient cohorts ($p_1, p_2$) or does it cause excessive slot turnover?
3. **Exact Free-Form Generation Accuracy**: How do the perplexity scores translate to exact MCQ primeAnswer accuracy on patient clinical diagnostic questions?

### LongHealth 5-Stage Patient Mapping (Sequential Cohorts)
Each stage covers 4 distinct patients and 80 multiple-choice clinical questions:
- `p01` (Stage 1): `patient_01`, `patient_02`, `patient_03`, `patient_04` (80 questions)
- `p02` (Stage 2): `patient_05`, `patient_06`, `patient_07`, `patient_08` (80 questions)
- `p03` (Stage 3): `patient_09`, `patient_10`, `patient_11`, `patient_12` (80 questions)
- `p04` (Stage 4): `patient_13`, `patient_14`, `patient_15`, `patient_16` (80 questions)
- `p05` (Stage 5): `patient_17`, `patient_18`, `patient_19`, `patient_20` (80 questions)

---

## 2. Environment Setup

### Step 1: Hugging Face Authentication
Ensure your Hugging Face token is set (required for accessing the raw LongHealth source dataset):

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

### Step 3: Pre-cache Qwen 3 4B Weights Offline
The evaluation driver operates under `HF_HUB_OFFLINE=1` for determinism. Pre-download the tokenizer and model weights before launching sweeps:

```bash
python3 -c '
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen3-4B-Instruct-2507"
print(f"Downloading {model_id}...")
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id)
print("Qwen 3 4B weights successfully cached locally!")
'
```

---

## 3. Dataset Hydration & Verification

### Step 1: Hydrate Dataset Artifacts
Hydrate base dataset shards from Hugging Face:
```bash
bash scripts/prepare_artifacts.sh --which data
```

### Step 2: Generate LongHealth Synth Parquets
The LongHealth self-study training parquets (`data/longhealth/synth/p0{1..5}/`) must be generated locally from the raw source to enforce the sequential patient split:

```bash
# Downloads raw LongHealth data from HF and builds sequential 5-stage synth parquets.
# Requires HF read access (HF_TOKEN is sufficient).
python3 examples/split_longhealth_stages.py \
  --no-upload \
  --keep-local data/longhealth
```

Verify that all 5 stages exist and contain sequential patients:
```bash
python3 -c "
import pyarrow.parquet as pq, re
from pathlib import Path
for stage in ['p01','p02','p03','p04','p05']:
    pq_path = Path(f'data/longhealth/synth/{stage}/self_study-n8192/artifact/dataset.parquet')
    if not pq_path.exists():
        print(f'{stage}: MISSING'); continue
    df = pq.read_table(pq_path).to_pandas()
    ids = set()
    for msg_list in df['messages']:
        for m in msg_list:
            ids.update(re.findall(r'patient_\d+', str(m.get('content', ''))))
    print(f'{stage}: {len(df)} rows | patients: {sorted(ids)}')
"
```

### Step 3: Prepare LongHealth Phase Eval Parquets
Materialize the 5-phase LongHealth eval parquets and train symlinks:
```bash
python3 examples/maintenance/data/prepare_longhealth_phases.py
```

### Step 4: Purge Any Stale LongHealth Chain Artifacts
> [!CAUTION]
> If any LongHealth chain artifacts exist from prior runs (e.g., using old interleaved patient mapping or different query counts), purge them before running the ablation to ensure a completely clean state:
```bash
rm -rf \
  outputs/experiments/subkv_sweep_longhealth/ \
  outputs/longhealth_5phase_state/ \
  outputs/longhealth_5phase_runs/ \
  outputs/evaluations/longhealth/
```

### Step 5: Run Pre-Flight Dataset Verification
Verify that all 5 eval phases and 5 training synth files exist for LongHealth:

```bash
python3 -c "
import sys
from examples.shared.am.continual_env import spec

s = spec('longhealth')
missing_eval = [p for p in range(1, 6) if not s.eval_path(p).exists()]
missing_synth = [p for p in range(1, 6) if not s.synth_path(p).exists()]

if missing_eval or missing_synth:
    print(f'[FAIL] longhealth:')
    if missing_eval: print(f'  Missing eval phases: {missing_eval}')
    if missing_synth: print(f'  Missing synth phases: {missing_synth}')
    sys.exit(1)
else:
    print('[OK] longhealth: all 5 eval and 5 synth parquets found.')
    print('Ready for ablation sweep execution!')
"
```

---

## 4. Execution Checklist (To-Do List)

Track your execution progress:

### Pre-Flight
- [ ] Export `HF_TOKEN`.
- [ ] Verify local Python environment and editable install (`pip install -e .`).
- [ ] Pre-cache Qwen 3 4B weights locally (`Qwen/Qwen3-4B-Instruct-2507`).
- [ ] Hydrate dataset artifacts (`bash scripts/prepare_artifacts.sh --which data`).
- [ ] Generate LongHealth synth parquets (`python3 examples/split_longhealth_stages.py --no-upload --keep-local data/longhealth`).
- [ ] Prepare LongHealth phase eval parquets (`python3 examples/maintenance/data/prepare_longhealth_phases.py`).
- [ ] Purge stale LongHealth artifacts (`rm -rf outputs/experiments/subkv_sweep_longhealth/ outputs/longhealth_5phase_state/ outputs/longhealth_5phase_runs/ outputs/evaluations/longhealth/`).
- [ ] Run pre-flight dataset verification script (`longhealth` `[OK]`).
- [ ] Run `--dry-run` for both Group A and Group B to verify recipes and plans.

---

### Group A: Baseline Regime (`q64`, $\lambda_\Delta = 0.01$)
- [ ] **Launch Group A Sweep**: Budget 512, top-t 32 and 64, max queries 64, delta weight 0.01 (`--tag-suffix _q64`).
- [ ] **Arm 1 (`qwen3_4b_budget512_topt32_q64`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/slot_frequency/summary.json`
- [ ] **Arm 2 (`qwen3_4b_budget512_topt64_q64`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/slot_frequency/summary.json`

---

### Group B: Scaled Regime (`q1024`, $\lambda_\Delta = 0.16$)
- [ ] **Launch Group B Sweep**: Budget 512, top-t 32 and 64, max queries 1024, delta weight 0.16 (`--tag-suffix _q1024`).
- [ ] **Arm 3 (`qwen3_4b_budget512_topt32_q1024`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/slot_frequency/summary.json`
- [ ] **Arm 4 (`qwen3_4b_budget512_topt64_q1024`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/slot_frequency/summary.json`

---

### Post-Run Comparative Analysis
- [ ] Compare diagonal acquisition accuracy across all 4 arms ($p_1 \dots p_5$).
- [ ] Compare off-diagonal retention accuracy ($p_1$ accuracy at stage $p_5$).
- [ ] Inspect slot overlap geometry differences between $t=32$ and $t=64$.
- [ ] Document final results in comparative summary table.

---

## 5. Execution Commands

### Strategy 1: Sequential Group Execution (Recommended for Single GPU)

You can launch using the convenience wrapper `run_sweep_qwen.sh` (or `run_sweep.sh`):

#### Batch 1: Run Group A (Baseline: $N_q=64, \lambda_\Delta=0.01$)
Executes Arm 1 ($t=32$) and Arm 2 ($t=64$) sequentially on GPU 0:

```bash
# Group A: Baseline (64 queries, delta_weight=0.01) on top-t 32 and 64
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0
```

#### Batch 2: Run Group B (Scaled: $N_q=1024, \lambda_\Delta=0.16$)
Executes Arm 3 ($t=32$) and Arm 4 ($t=64$) sequentially on GPU 0:

```bash
# Group B: Scaled (1024 queries, delta_weight=0.16) on top-t 32 and 64
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 1024 \
  --delta-weight 0.16 \
  --tag-suffix _q1024 \
  --eval-accuracy \
  --gpu 0
```

---

### Strategy 2: Individual Per-Arm Commands (Parallel Execution across GPUs)

If multiple GPUs are available, you can run all 4 arms concurrently:

#### Arm 1 (GPU 0): Budget 512, top-t 32, $N_q=64, \lambda_\Delta=0.01$
```bash
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-t 32 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0
```

#### Arm 2 (GPU 1): Budget 512, top-t 64, $N_q=64, \lambda_\Delta=0.01$
```bash
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-t 64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 1
```

#### Arm 3 (GPU 2): Budget 512, top-t 32, $N_q=1024, \lambda_\Delta=0.16$
```bash
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-t 32 \
  --max-queries-per-head 1024 \
  --delta-weight 0.16 \
  --tag-suffix _q1024 \
  --eval-accuracy \
  --gpu 2
```

#### Arm 4 (GPU 3): Budget 512, top-t 64, $N_q=1024, \lambda_\Delta=0.16$
```bash
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-t 64 \
  --max-queries-per-head 1024 \
  --delta-weight 0.16 \
  --tag-suffix _q1024 \
  --eval-accuracy \
  --gpu 3
```

---

### Strategy 3: Multi-GPU Evaluation Acceleration

To accelerate the held-out MCQ generation accuracy evaluations across all available GPUs:

```bash
# Run Group A with evaluation fanned out across GPUs 0,1,2,3
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0 \
  --eval-gpus 0,1,2,3

# Run Group B with evaluation fanned out across GPUs 0,1,2,3
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 1024 \
  --delta-weight 0.16 \
  --tag-suffix _q1024 \
  --eval-accuracy \
  --gpu 0 \
  --eval-gpus 0,1,2,3
```

---

### Strategy 4: Dry-Run Verification

Always test recipe generation and execution plans before committing GPU compute:

```bash
# Dry-run Group A
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --dry-run

# Dry-run Group B
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 1024 \
  --delta-weight 0.16 \
  --tag-suffix _q1024 \
  --dry-run
```

---

### Strategy 5: Running in Background (`tmux` / `nohup`)

#### Using `tmux` (Recommended)
```bash
tmux new -s qwen_ablation

# Inside tmux:
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0

# Detach: Ctrl+B then D
# Re-attach:
tmux attach -t qwen_ablation
```

#### Using `nohup`
```bash
nohup bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0 > logs_group_a.out 2>&1 &

tail -f logs_group_a.out
```

---

## 6. Output Artifacts & Result Inspection

### Result File Locations

All 4 arms produce isolated output directories matching the `qwen3_4b_budget...` structure:

| Arm | Tag Identifier | Accuracy Matrix Path | Log-Perplexity Matrix Path |
| :--- | :--- | :--- | :--- |
| **Arm 1** | `qwen3_4b_budget512_topt32_q64` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 2** | `qwen3_4b_budget512_topt64_q64` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 3** | `qwen3_4b_budget512_topt32_q1024` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 4** | `qwen3_4b_budget512_topt64_q1024` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/teacher-forced-logppl-v1/matrix.csv` |

### Quick Commands to View Accuracy Matrices
```bash
# Arm 1: top_t 32, q64
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# Arm 2: top_t 64, q64
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# Arm 3: top_t 32, q1024
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# Arm 4: top_t 64, q1024
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv
```

### Quick Commands to View Slot Overlap & Retention Metrics
```bash
# View Jaccard overlap matrices across stages:
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/slot_frequency/jaccard_overlap.csv
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/slot_frequency/jaccard_overlap.csv

# Inspect summary JSON:
cat outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/slot_frequency/summary.json
cat outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/slot_frequency/summary.json
```

---

## 7. Comparative Ablation Analysis Template

Once all 4 runs complete, populate this comparative summary table to evaluate the ablation results:

| Arm | Config ($S / t / N_q / \lambda_\Delta$) | Avg Initial Acquisition $(p_i, p_i)$ | Final Task 1 Retention $(p_5, p_1)$ | Task 1 Drop $(p_1, p_1) - (p_5, p_1)$ | Mean Matrix Accuracy | Mean Matrix Log-Perplexity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | $512 / 32 / 64 / 0.01$ | | | | | |
| **Arm 2** | $512 / 64 / 64 / 0.01$ | | | | | |
| **Arm 3** | $512 / 32 / 1024 / 0.16$ | | | | | |
| **Arm 4** | $512 / 64 / 1024 / 0.16$ | | | | | |

- **Acquisition Hypothesis**: Scaled queries ($N_q=1024, \lambda_\Delta=0.16$) should yield higher diagonal acquisition $(p_i, p_i)$ due to richer gradient-free AM target matching.
- **Retention Hypothesis**: Doubling the update budget to $t=64$ may improve later-stage acquisition at the cost of higher slot overwriting on early patient cohorts ($p_1, p_2$). Compare $(p_5, p_1)$ between $t=32$ and $t=64$ to test this trade-off.
