# Experiment 1: Cartridge-Conditioned Self-Study Data Generation

## Hypothesis

Does injecting a trained cartridge (from year Y) during self-study synthesis for a
**later year (Y+1)** produce better quality training data than synthesis without a cartridge?

The cartridge encodes compressed prior-year knowledge as a KV-cache prefix. When injected
during synthesis of the next year's document, the model can draw on that background context
to generate more grounded and meaningful Q&A pairs — without needing the prior document in
the prompt.

**Comparison:**
- Baseline: self-study synthesis on year Y+1 document, no cartridge
- Experiment: self-study synthesis on year Y+1 document, with year-Y cartridge injected

---

## Pipeline Overview

```
Year-Y document
     │
     ▼
[Step 1] Synthesize self-study data on year Y (base model, no cartridge)
     │
     ▼
[Step 2] Train cartridge on year-Y synthesized data
     │
     ▼
[Step 3] Upload year-Y cartridge to HuggingFace
     │
     ▼                                    ┌─────────────────────────────┐
[Step 4a] Synthesize self-study data      │ [Step 4b] Synthesize with   │
          on year Y+1 (no cartridge)      │ year-Y cartridge injected   │
          ← baseline                      │ ← experiment                │
                                          └─────────────────────────────┘
     │                                           │
     └──────────────── Compare quality ──────────┘
```

---

## Prerequisites

- Tokasaurus server running (default: `http://localhost:8000`)
- `CARTRIDGES_TOKASAURUS_URL` env var if not using localhost
- `CARTRIDGES_OUTPUT_DIR` env var for output location (default: `.`)

### Starting the Tokasaurus Server

> For full installation and configuration options, see the [Tokasaurus documentation](https://github.com/ScalingIntelligence/tokasaurus).

```bash
tksrs \
    model=<MODEL_ID> \
    kv_cache_num_tokens='(512 * 1024)' \
    max_topk_logprobs=20 \
    dp_size=<NUM_GPUS>
```

| Parameter | Example | Description |
|---|---|---|
| `model` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID to serve |
| `kv_cache_num_tokens` | `'(512 * 1024)'` | Total KV cache token budget |
| `max_topk_logprobs` | `20` | Max top-k logprobs returned per token |
| `dp_size` | `1` | Number of data-parallel replicas (= number of GPUs) |

---

## Step 1 — Synthesize self-study data on year Y (no cartridge)

Generate Q&A training data from the year-Y document using the base model only.

```bash
python experiments/experiments_1/synthesize_self_study_data.py \
    --company <COMPANY> --year <YEAR_Y> \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --num_samples 64 --batch_size 4 --max_num_batches 4
```

Output: `$CARTRIDGES_OUTPUT_DIR/synthesize_<company>_<year>_.../dataset_clean.parquet`

---

## Step 2 — Train the cartridge on year Y

Train a KV-cache cartridge on the Step 1 synthesized data.

```bash
SYNTH_DATA_PATH_PHASE1=/path/to/step1/dataset_clean.parquet \
python experiments/experiments_1/train_initial.py
```

Optional env vars:
| Variable | Default | Description |
|---|---|---|
| `SYNTH_DATA_PATH_PHASE1` | *(required)* | Path to Step 1 parquet |
| `TEXT_PATH` | `data/texts/AMD_2021_10K.txt` | Source 10-K text used for KV init |
| `NUM_TOKENS` | `4096` | Cartridge capacity in tokens |

Output: `$CARTRIDGES_OUTPUT_DIR/<timestamp>-train_initial/<uuid>/cache-step<N>.pt`
and `config.yaml` in the same directory.

---

## Step 3 — Upload year-Y cartridge to HuggingFace

```bash
python experiments/experiments_1/upload_cartridge_to_hf.py \
    --cartridge-path outputs/<timestamp>-train_initial/<uuid>/cache-step<N>.pt \
    --hf-repo-id <hf-username>/<repo-name>
```

Uploads: `.pt` weights + `config.yaml` + model card `README.md`.

---

## Step 4a — Synthesize self-study data on year Y+1 (baseline, no cartridge)

```bash
python experiments/experiments_1/synthesize_self_study_data.py \
    --company <COMPANY> --year <YEAR_Y+1> \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --num_samples 64 --batch_size 4 --max_num_batches 4
```

---

## Step 4b — Synthesize self-study data on year Y+1 (with year-Y cartridge)

```bash
python experiments/experiments_1/synthesize_self_study_data_with_cartridge.py \
    --company <COMPANY> --year <YEAR_Y+1> \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --num_samples 64 --batch_size 4 --max_num_batches 4 \
    --cartridge-hf-id <hf-username>/<repo-name>
```

When the cartridge is active you should see in the Tokasaurus server logs:
```
Loaded cartridge config: <hf-repo-id> with 4096 tokens from config.yaml
Loaded cartridge: <hf-repo-id>
```

---

## Experiment 1 Run (AMD 2021 → 2022)

The first run of this pipeline used:
- Year Y: AMD 2021 10-K
- Year Y+1: AMD 2022 10-K
- Cartridge: `Phudish/amd-2021-cartridge-without-year-experiment-1` (4096 tokens)
- Year excluded from prompts (`--include-year` not set)
