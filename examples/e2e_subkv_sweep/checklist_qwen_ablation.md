# Qwen 3 4B Continual Compaction: LongHealth & QuALITY Ablation Checklist & Execution Handoff

This document provides a complete checklist, environment setup guide, dataset hydration steps, and actionable execution commands for running the **4-Arm Ablation Study** on **Qwen 3 4B Instruct** (`Qwen/Qwen3-4B-Instruct-2507`) across both **LongHealth** and **QuALITY** benchmarks across 5 sequential stages, evaluating both **Multiple-Choice Option Generation Accuracy** (`accuracy-freeform-mc-options-primeAnswer-v1` 5×5 matrix) and **Teacher-Forced Log-Perplexity** (`teacher-forced-logppl-v1` 5×5 matrix).

---

## 1. Overview & Experimental Configuration

This ablation evaluates continual memory retention and acquisition on **Qwen 3 4B** across two distinct long-context reasoning domains:
1. **LongHealth**: 5 sequential clinical patient cohorts (4 patients / 80 questions per stage, 400 questions total, 5 options `(a)-(e)`).
2. **QuALITY**: 5 sequential long-narrative fiction cohorts (7 articles / 122–128 questions per stage, 622 questions total, 4 options `(a)-(d)`).

All 4 runs use a fixed sub-KV cache budget $S = 512$, sweeping over update budgets $\text{top\_t} \in \{32, 64\}$ across two query ceiling and delta regularization regimes:
1. **Group A (Baseline Regime)**: `max_queries_per_head = 64`, `delta_weight = 0.01` ($\lambda_\Delta = 0.01$).
2. **Group B (Scaled Regime)**: `max_queries_per_head = 1024`, `delta_weight = 0.16` ($\lambda_\Delta = 0.01 \times \frac{1024}{64} = 0.16$).

### 4-Arm Matrix Summary (Identical for LongHealth & QuALITY)

| Arm | Sub-KV Budget ($S$) | Update Budget $\text{top\_t}$ ($t$) | Update Ratio ($t / S$) | Max Queries ($N_q$) | Regularization ($\lambda_\Delta$) | Tag Suffix | Lineage / Tag Identifier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arm 1 (A1)** | **512** | **32** | 6.25% | **64** | **0.01** | `_q64` | `qwen3_4b_budget512_topt32_q64` |
| **Arm 2 (A2)** | **512** | **64** | 12.50% | **64** | **0.01** | `_q64` | `qwen3_4b_budget512_topt64_q64` |
| **Arm 3 (B1)** | **512** | **32** | 6.25% | **1024** | **0.16** | `_q1024` | `qwen3_4b_budget512_topt32_q1024` |
| **Arm 4 (B2)** | **512** | **64** | 12.50% | **1024** | **0.16** | `_q1024` | `qwen3_4b_budget512_topt64_q1024` |

### Unified MCQ Evaluation Protocol (`accuracy-freeform-mc-options-primeAnswer-v1`)

Both LongHealth and QuALITY use the **exact same evaluation protocol** for direct generation accuracy:
- **No CoT / Reasoning**: Step-by-step thinking is disabled (`enable_thinking=False`). The model is prompted to directly output the answer letter along with the option content.
- **Answer Prime**: Generation is primed with ` Answer:` so the model directly begins decoding the final answer.
- **Token Budget**: `max_new_tokens=32` at `temperature=0.0` (greedy decoding). 32 tokens is fast and sufficient for the letter and option text (e.g. `(b) Rituximab` or `(a) Polatuzumab vedotin`).
- **Standardized Options**:
  - **LongHealth**: 5 lettered options `(a)` through `(e)`.
  - **QuALITY**: 4 lettered options `(a)` through `(d)`.
- **Scoring**: Evaluated by the robust `mc_options` extractor (letter matching $\to$ substring matching $\to$ fuzzy token matching).

### Stage Cohort Mappings

#### 1. LongHealth 5-Stage Patient Mapping (Sequential Cohorts)
Each stage covers 4 distinct patients and 80 multiple-choice clinical questions:
- `p01` (Stage 1): `patient_01`, `patient_02`, `patient_03`, `patient_04` (80 questions)
- `p02` (Stage 2): `patient_05`, `patient_06`, `patient_07`, `patient_08` (80 questions)
- `p03` (Stage 3): `patient_09`, `patient_10`, `patient_11`, `patient_12` (80 questions)
- `p04` (Stage 4): `patient_13`, `patient_14`, `patient_15`, `patient_16` (80 questions)
- `p05` (Stage 5): `patient_17`, `patient_18`, `patient_19`, `patient_20` (80 questions)
- **Total**: 20 patients, 400 clinical questions.

#### 2. QuALITY 5-Stage Narrative Mapping (Sequential Articles)
Each stage covers 7 long narrative science fiction articles (7,442 to 8,445 tokens per article, ~55k tokens per stage):
- `p01` (Stage 1): 7 articles (54,708 tokens, 124 questions)
- `p02` (Stage 2): 7 articles (54,720 tokens, 128 questions)
- `p03` (Stage 3): 7 articles (54,716 tokens, 122 questions)
- `p04` (Stage 4): 7 articles (54,701 tokens, 126 questions)
- `p05` (Stage 5): 7 articles (54,703 tokens, 122 questions)
- **Total**: 35 articles, 273,548 tokens, 622 questions.

---

## 2. Environment Setup

### Step 1: Hugging Face Authentication
Ensure your Hugging Face token is set (required for accessing source datasets and artifacts):

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
Hydrate base dataset shards from Hugging Face (this automatically downloads QuALITY training synth parquets and base files):
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

Verify that all 5 LongHealth stages exist and contain sequential patients:
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
> [!IMPORTANT]
> The LongHealth evaluation pipeline has been updated to use the standardized direct lettered format `(a)-(e)` with no CoT and `max_new_tokens=32`, matching QuALITY.
> Running `prepare_longhealth_phases.py` reconstructs `data/longhealth/phases/phase{1..5}_eval.parquet` with this updated format:

```bash
python3 examples/maintenance/data/prepare_longhealth_phases.py
```

Verify the updated prompt format in LongHealth eval parquets:
```bash
python3 -c "
import pyarrow.parquet as pq
df = pq.read_table('data/longhealth/phases/phase1_eval.parquet').to_pandas()
print('Phase 1 eval rows:', len(df))
print('Sample prompt:')
print(df.iloc[0]['messages'][0]['content'][-300:])
"
```

### Step 4: Verify QuALITY Dataset Parquets
QuALITY eval and synth parquets are located in `data/quality/`. Verify that all 5 eval parquets and synth directories are present:

```bash
python3 -c "
import pyarrow.parquet as pq
from pathlib import Path
for phase in range(1, 6):
    eval_p = Path(f'data/quality/phases/phase{phase}_eval.parquet')
    synth_p = Path(f'data/quality/synth/p0{phase}/self_study-n8192/artifact/dataset.parquet')
    print(f'Stage p0{phase}: eval exists={eval_p.exists()} (rows={len(pq.read_table(eval_p)) if eval_p.exists() else 0}) | synth exists={synth_p.exists()}')
"
```

### Step 5: Purge Any Stale Artifacts
> [!CAUTION]
> If any artifacts exist from prior runs (e.g. older LongHealth runs with CoT prompts or old QuALITY runs), purge them before running the sweeps to guarantee clean, uncorrupted state:

```bash
# Purge stale LongHealth artifacts
rm -rf \
  outputs/experiments/subkv_sweep_longhealth/ \
  outputs/longhealth_5phase_state/ \
  outputs/longhealth_5phase_runs/ \
  outputs/evaluations/longhealth/

# Purge stale QuALITY artifacts
rm -rf \
  outputs/experiments/subkv_sweep_quality/ \
  outputs/quality_5phase_state/ \
  outputs/quality_5phase_runs/ \
  outputs/evaluations/quality/
```

### Step 6: Run Pre-Flight Dataset Verification
Verify that both `longhealth` and `quality` have all 5 eval phases and 5 synth parquets available:

```bash
python3 -c "
import sys
from examples.shared.am.continual_env import spec

for ds in ['longhealth', 'quality']:
    s = spec(ds)
    missing_eval = [p for p in range(1, 6) if not s.eval_path(p).exists()]
    missing_synth = [p for p in range(1, 6) if not s.synth_path(p).exists()]
    if missing_eval or missing_synth:
        print(f'[FAIL] {ds}:')
        if missing_eval: print(f'  Missing eval phases: {missing_eval}')
        if missing_synth: print(f'  Missing synth phases: {missing_synth}')
        sys.exit(1)
    else:
        print(f'[OK] {ds}: all 5 eval and 5 synth parquets verified.')

print('\nAll datasets ready for ablation sweep execution!')
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
- [ ] Verify QuALITY phase eval and synth parquets.
- [ ] Purge stale LongHealth and QuALITY artifacts.
- [ ] Run pre-flight verification script (`longhealth` and `quality` both `[OK]`).
- [ ] Run `--dry-run` for both Group A and Group B across both datasets.

---

### Part 1: LongHealth 4-Arm Sweep (Re-Run with New Non-CoT Protocol)

#### Group A: Baseline Regime (`q64`, $\lambda_\Delta = 0.01$)
- [ ] **Launch LongHealth Group A Sweep**: Budget 512, top-t 32 and 64, max queries 64, delta weight 0.01 (`--tag-suffix _q64`).
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

#### Group B: Scaled Regime (`q1024`, $\lambda_\Delta = 0.16$)
- [ ] **Launch LongHealth Group B Sweep**: Budget 512, top-t 32 and 64, max queries 1024, delta weight 0.16 (`--tag-suffix _q1024`).
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

### Part 2: QuALITY 4-Arm Sweep (E2E-Train & Eval on Narrative Reading)

#### Group A: Baseline Regime (`q64`, $\lambda_\Delta = 0.01$)
- [ ] **Launch QuALITY Group A Sweep**: Budget 512, top-t 32 and 64, max queries 64, delta weight 0.01 (`--tag-suffix _q64`).
- [ ] **Arm 1 (`qwen3_4b_budget512_topt32_q64`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/slot_frequency/summary.json`
- [ ] **Arm 2 (`qwen3_4b_budget512_topt64_q64`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q64/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q64/slot_frequency/summary.json`

#### Group B: Scaled Regime (`q1024`, $\lambda_\Delta = 0.16$)
- [ ] **Launch QuALITY Group B Sweep**: Budget 512, top-t 32 and 64, max queries 1024, delta weight 0.16 (`--tag-suffix _q1024`).
- [ ] **Arm 3 (`qwen3_4b_budget512_topt32_q1024`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/slot_frequency/summary.json`
- [ ] **Arm 4 (`qwen3_4b_budget512_topt64_q1024`) Results**:
  - [ ] Verify logppl: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q1024/teacher-forced-logppl-v1/matrix.csv`
  - [ ] Verify accuracy: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv`
  - [ ] Verify generations: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/cells/*/generations.jsonl`
  - [ ] Verify slot frequency: `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q1024/slot_frequency/summary.json`

---

### Post-Run Comparative Analysis
- [ ] Compare diagonal acquisition accuracy across all 4 arms for LongHealth.
- [ ] Compare diagonal acquisition accuracy across all 4 arms for QuALITY.
- [ ] Compare off-diagonal retention accuracy ($p_1$ retention at stage $p_5$) for LongHealth and QuALITY.
- [ ] Cross-compare clinical vs. narrative domain sensitivity to $t=32$ vs $t=64$.
- [ ] Document final results in comparative summary tables.

---

## 5. Execution Commands

### Strategy 1: Sequential Group Execution (Single GPU)

Run each group sequentially on a single GPU. The `--eval-accuracy` flag automatically generates both the perplexity and direct MCQ accuracy matrices:

#### Option A: Run LongHealth Followed by QuALITY

```bash
# -----------------------------------------------------------------------------
# 1. LongHealth Group A: Baseline (64 queries, delta_weight=0.01)
# -----------------------------------------------------------------------------
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0

# -----------------------------------------------------------------------------
# 2. LongHealth Group B: Scaled (1024 queries, delta_weight=0.16)
# -----------------------------------------------------------------------------
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 1024 \
  --delta-weight 0.16 \
  --tag-suffix _q1024 \
  --eval-accuracy \
  --gpu 0

# -----------------------------------------------------------------------------
# 3. QuALITY Group A: Baseline (64 queries, delta_weight=0.01)
# -----------------------------------------------------------------------------
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0

# -----------------------------------------------------------------------------
# 4. QuALITY Group B: Scaled (1024 queries, delta_weight=0.16)
# -----------------------------------------------------------------------------
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 1024 \
  --delta-weight 0.16 \
  --tag-suffix _q1024 \
  --eval-accuracy \
  --gpu 0
```

#### Option B: Joint Multi-Dataset Execution (`--datasets longhealth,quality`)
You can pass both datasets in a single command; the sweep driver will run LongHealth then QuALITY end-to-end:

```bash
# Joint Group A: LongHealth & QuALITY
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --datasets longhealth,quality \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0

# Joint Group B: LongHealth & QuALITY
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --datasets longhealth,quality \
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

If you have 4 GPUs available (e.g. `GPU 0, 1, 2, 3`), you can run all 4 arms concurrently.

#### LongHealth Parallel Execution (GPUs 0–3)
```bash
# Arm 1 (GPU 0): Budget 512, top-t 32, Nq=64, delta=0.01
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth --budget 512 --top-t 32 --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 --eval-accuracy --gpu 0 &

# Arm 2 (GPU 1): Budget 512, top-t 64, Nq=64, delta=0.01
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth --budget 512 --top-t 64 --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 --eval-accuracy --gpu 1 &

# Arm 3 (GPU 2): Budget 512, top-t 32, Nq=1024, delta=0.16
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth --budget 512 --top-t 32 --max-queries-per-head 1024 --delta-weight 0.16 --tag-suffix _q1024 --eval-accuracy --gpu 2 &

# Arm 4 (GPU 3): Budget 512, top-t 64, Nq=1024, delta=0.16
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth --budget 512 --top-t 64 --max-queries-per-head 1024 --delta-weight 0.16 --tag-suffix _q1024 --eval-accuracy --gpu 3 &

wait
```

#### QuALITY Parallel Execution (GPUs 0–3)
```bash
# Arm 1 (GPU 0): Budget 512, top-t 32, Nq=64, delta=0.01
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality --budget 512 --top-t 32 --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 --eval-accuracy --gpu 0 &

# Arm 2 (GPU 1): Budget 512, top-t 64, Nq=64, delta=0.01
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality --budget 512 --top-t 64 --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 --eval-accuracy --gpu 1 &

# Arm 3 (GPU 2): Budget 512, top-t 32, Nq=1024, delta=0.16
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality --budget 512 --top-t 32 --max-queries-per-head 1024 --delta-weight 0.16 --tag-suffix _q1024 --eval-accuracy --gpu 2 &

# Arm 4 (GPU 3): Budget 512, top-t 64, Nq=1024, delta=0.16
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality --budget 512 --top-t 64 --max-queries-per-head 1024 --delta-weight 0.16 --tag-suffix _q1024 --eval-accuracy --gpu 3 &

wait
```

---

### Strategy 3: Multi-GPU Evaluation Acceleration

When running sequential compaction training on GPU 0, you can accelerate the held-out MCQ generation evaluation by fanning the 5 evaluation stages across all GPUs (`--eval-gpus 0,1,2,3`):

```bash
# LongHealth Group A with multi-GPU evaluation fanout:
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth --budget 512 --top-ts 32,64 \
  --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 \
  --eval-accuracy --gpu 0 --eval-gpus 0,1,2,3

# QuALITY Group A with multi-GPU evaluation fanout:
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality --budget 512 --top-ts 32,64 \
  --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 \
  --eval-accuracy --gpu 0 --eval-gpus 0,1,2,3
```

---

### Strategy 4: Dry-Run Verification

Always test recipe generation and execution plans before committing GPU compute:

```bash
# LongHealth Dry-run
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset longhealth --budget 512 --top-ts 32,64 \
  --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 --dry-run

# QuALITY Dry-run
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --dataset quality --budget 512 --top-ts 32,64 \
  --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 --dry-run
```

---

### Strategy 5: Running in Background (`tmux` / `nohup`)

#### Using `tmux` (Recommended)
```bash
tmux new -s qwen_sweep

# Inside tmux, launch LongHealth + QuALITY:
bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --datasets longhealth,quality \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0

# Detach: Ctrl+B then D
# Re-attach at any time:
tmux attach -t qwen_sweep
```

#### Using `nohup`
```bash
nohup bash examples/e2e_subkv_sweep/run_sweep_qwen.sh \
  --datasets longhealth,quality \
  --budget 512 \
  --top-ts 32,64 \
  --max-queries-per-head 64 \
  --delta-weight 0.01 \
  --tag-suffix _q64 \
  --eval-accuracy \
  --gpu 0 > sweep_group_a.out 2>&1 &

tail -f sweep_group_a.out
```

---

## 6. Output Artifacts & Result Inspection

### Result File Locations

Both datasets produce isolated output directories matching the `qwen3_4b_budget...` lineage:

#### LongHealth Artifacts (`outputs/evaluations/longhealth/<TAG>/...`)
| Arm | Tag Identifier | Accuracy Matrix Path | Log-Perplexity Matrix Path |
| :--- | :--- | :--- | :--- |
| **Arm 1** | `qwen3_4b_budget512_topt32_q64` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 2** | `qwen3_4b_budget512_topt64_q64` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 3** | `qwen3_4b_budget512_topt32_q1024` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 4** | `qwen3_4b_budget512_topt64_q1024` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/teacher-forced-logppl-v1/matrix.csv` |

#### QuALITY Artifacts (`outputs/evaluations/quality/<TAG>/...`)
| Arm | Tag Identifier | Accuracy Matrix Path | Log-Perplexity Matrix Path |
| :--- | :--- | :--- | :--- |
| **Arm 1** | `qwen3_4b_budget512_topt32_q64` | `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 2** | `qwen3_4b_budget512_topt64_q64` | `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q64/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 3** | `qwen3_4b_budget512_topt32_q1024` | `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/teacher-forced-logppl-v1/matrix.csv` |
| **Arm 4** | `qwen3_4b_budget512_topt64_q1024` | `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv` | `outputs/evaluations/quality/qwen3_4b_budget512_topt64_q1024/teacher-forced-logppl-v1/matrix.csv` |

### Quick Commands to View Accuracy Matrices

#### LongHealth
```bash
# LongHealth Arm 1: top_t 32, q64
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# LongHealth Arm 2: top_t 64, q64
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# LongHealth Arm 3: top_t 32, q1024
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# LongHealth Arm 4: top_t 64, q1024
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv
```

#### QuALITY
```bash
# QuALITY Arm 1: top_t 32, q64
column -s, -t outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# QuALITY Arm 2: top_t 64, q64
column -s, -t outputs/evaluations/quality/qwen3_4b_budget512_topt64_q64/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# QuALITY Arm 3: top_t 32, q1024
column -s, -t outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv

# QuALITY Arm 4: top_t 64, q1024
column -s, -t outputs/evaluations/quality/qwen3_4b_budget512_topt64_q1024/accuracy-freeform-mc-options-primeAnswer-v1/matrix.csv
```

### Quick Commands to View Slot Overlap & Retention Metrics
```bash
# LongHealth Jaccard overlap matrices:
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q64/slot_frequency/jaccard_overlap.csv
column -s, -t outputs/evaluations/longhealth/qwen3_4b_budget512_topt32_q1024/slot_frequency/jaccard_overlap.csv

# QuALITY Jaccard overlap matrices:
column -s, -t outputs/evaluations/quality/qwen3_4b_budget512_topt32_q64/slot_frequency/jaccard_overlap.csv
column -s, -t outputs/evaluations/quality/qwen3_4b_budget512_topt32_q1024/slot_frequency/jaccard_overlap.csv
```

---

## 7. Comparative Ablation Analysis Template

Once both sweeps complete, populate the summary tables below to evaluate the ablation results:

### Table 1: LongHealth 4-Arm Ablation Results (Clinical EHR Domain)

| Arm | Config ($S / t / N_q / \lambda_\Delta$) | Avg Initial Acquisition $(p_i, p_i)$ | Final Task 1 Retention $(p_5, p_1)$ | Task 1 Drop $(p_1, p_1) - (p_5, p_1)$ | Mean Matrix Accuracy | Mean Matrix Log-Perplexity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | $512 / 32 / 64 / 0.01$ | | | | | |
| **Arm 2** | $512 / 64 / 64 / 0.01$ | | | | | |
| **Arm 3** | $512 / 32 / 1024 / 0.16$ | | | | | |
| **Arm 4** | $512 / 64 / 1024 / 0.16$ | | | | | |

### Table 2: QuALITY 4-Arm Ablation Results (Long-Context Narrative Domain)

| Arm | Config ($S / t / N_q / \lambda_\Delta$) | Avg Initial Acquisition $(p_i, p_i)$ | Final Task 1 Retention $(p_5, p_1)$ | Task 1 Drop $(p_1, p_1) - (p_5, p_1)$ | Mean Matrix Accuracy | Mean Matrix Log-Perplexity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | $512 / 32 / 64 / 0.01$ | | | | | |
| **Arm 2** | $512 / 64 / 64 / 0.01$ | | | | | |
| **Arm 3** | $512 / 32 / 1024 / 0.16$ | | | | | |
| **Arm 4** | $512 / 64 / 1024 / 0.16$ | | | | | |

### Key Scientific Hypotheses & Cross-Domain Analysis

1. **Acquisition Hypothesis**:
   - Scaled reference queries ($N_q=1024, \lambda_\Delta=0.16$) should yield higher diagonal acquisition $(p_i, p_i)$ across both datasets due to richer gradient-free AM target matching over long sequences.
2. **Retention vs. Overwrite Trade-off**:
   - Doubling the update budget to $t=64$ (12.5% update ratio) allows faster adaptation to new phases, but may accelerate slot overwriting on earlier cohorts ($p_1, p_2$). Compare $(p_5, p_1)$ between $t=32$ and $t=64$ to quantify whether retention suffers more in clinical records vs. literary narratives.
3. **Cross-Domain Consistency**:
   - Compare whether the optimal configuration ($t=32$ vs $t=64$, $N_q=64$ vs $N_q=1024$) is consistent across LongHealth (fact-dense clinical EHRs) and QuALITY (literary narrative fiction), or if domain-specific memory consolidation strategies emerge.
