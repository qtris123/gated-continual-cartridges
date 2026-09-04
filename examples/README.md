# Workflow examples

`examples/` is organized by responsibility:

- `shared/` contains Python workflow engines grouped by capability, including
  the continual-AM endpoints (`build_p01`, `run_chain`, `grad_step`) and the
  experiment manifests under `shared/am/manifests/`.
- `qasper/`, `quality/`, `longhealth/`, `finqa/`, and `techqa/` contain
  dataset-owned assets: synthesize recipes, benchmarks, and any dataset-specific
  helpers. Experiment workloads themselves live in manifests, not per-dataset
  wrapper scripts.
- `maintenance/` contains explicit one-off migration and data conversion tools.

Dataset recipes intentionally repeat environment variables, paths, and GPU
settings. They are executable run cards, not reusable libraries. Shared behavior
belongs in `examples/shared/` or, when it is part of the product API, in
`cartridges/`.

Datasets, prompt corpora, generated parquet files, and reference CSVs belong
under `data/<dataset>/`. Notebooks belong under `notebook/`; experiment reports
belong under `notes/`.

## Experiment runbook (example commands)

Run everything from the repository root. `PY=.venv/bin/python` is assumed; the
eval scripts also honor `CARTRIDGES_PYTHON` and `MODEL_NAME`
(default `Qwen/Qwen3-4B-Instruct-2507`).

### 0. Fetch artifacts first (data, recipes, caches)

Most drivers read inputs from `data/` and `outputs/` — populate them from
HuggingFace before running anything (see `scripts/hf/README.md`):

```bash
export HF_TOKEN=hf_...
scripts/prepare_artifacts.sh --which data        # datasets + synth
scripts/prepare_artifacts.sh --which results     # recipes, eval plans, prior results/plots
scripts/prepare_artifacts.sh --which caches      # optional: 132 GB compacted KV caches
```

Recipe configs (`outputs/recipes/*.yaml`) are tracked in git, so you can run
experiments without downloading caches.

### Axis A — synthesize self-study training data

```bash
bash examples/<dataset>/synthesize/self_study.sh   # dataset: qasper|quality|finqa|techqa|longhealth
```

### Axis B — continual-AM 5-phase compaction

```bash
# 1) Build the phase-1 (p01) cache for a dataset:
$PY examples/shared/am/build_p01.py \
  --dataset qasper --recipe-config outputs/recipes/topt64.yaml --gpu 0

# 2) Chain one cartridge through phases 2-5 and emit the 5x5 logppl grid:
$PY examples/shared/am/run_chain.py \
  --dataset quality --p01-cache outputs/hf_p01/quality/cache_last.pt \
  --tag soft_l1_v1 --recipe-config outputs/recipes/soft_locality_l1.yaml \
  --gpu 0 --eval-gpus 0,1,2,3
# (add --preflight to validate config/paths without running; chains are stage-resumable by --tag)

# 3) All four benchmarks, full-KV, one GPU each, in parallel:
bash examples/shared/am/run_all4_fullkv.sh
```

### Sweeps (manifest-driven — the main way to run arm families)

Manifests live in `examples/shared/am/manifests/`:
`soft_locality` (streams `top_t`, `lambda`), `capacity_top_t` (`top_t`),
`slots_per_doc`, `p01_rope`, `p01_armD`.

```bash
M=examples/shared/am/manifests/soft_locality.yaml

$PY examples/shared/am/sweep.py check  --manifest $M                       # validate manifest (no GPU)
$PY examples/shared/am/sweep.py arms   --manifest $M --stream lambda       # list resolved arms + tags
$PY examples/shared/am/sweep.py render --manifest $M                       # write outputs/recipes/<tag>.yaml per arm
$PY examples/shared/am/sweep.py launch --manifest $M --stream lambda --dataset quality --gpus 0,1,2,3
$PY examples/shared/am/sweep.py eval   --manifest $M --stream lambda --dataset quality --kind accuracy
```

### Evaluation (per lineage tag)

The teacher-forced logppl matrix (`matrix.json`) is emitted by `run_chain.py`.
For accuracy and free-form generations:

```bash
# accuracy matrix (freeform-mc-options primeAnswer), 5 stages fanned across GPUs:
bash examples/shared/evaluate/run_accuracy.sh   quality soft_l1_v1 0,1,2,3
# record autoregressive generations (generations.jsonl):
bash examples/shared/evaluate/record_generations.sh techqa topt64_v1 0,1,2,3
```

### Analysis / figures / tables

```bash
# per-stage slot-update geometry for a sweep stream:
$PY examples/shared/am/slot_geometry.py --manifest $M --stream lambda --dataset quality --which value
# forgetting / drift / structure / repetition (see --help for each):
$PY examples/shared/evaluate/forgetting.py --help
$PY examples/shared/evaluate/continual_cache_drift.py --help
# soft-locality report figures:
$PY outputs/experiments/soft_locality/report_figs/_make_figs.py
```

> Caveat: `run_chain.py`'s built-in `--recipe-config` default points at a
> config nested inside a compacted-cache run dir; always pass an explicit
> `--recipe-config outputs/recipes/<recipe>.yaml` unless you've fetched caches.
