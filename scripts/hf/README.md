# HuggingFace artifacts: setup & prepare

This repo's code lives in git; the **large artifacts** (datasets, compacted KV
caches, evaluation results/plots) live on HuggingFace and are pulled down with
one script. This directory contains the tooling to **download + set up** those
artifacts on a fresh checkout, and to **rebuild + re-upload** them.

## The three datasets

| Dataset | Size | Contents | Lands under |
| --- | --- | --- | --- |
| [`qtris123/gated-continual-cartridges-data`](https://huggingface.co/datasets/qtris123/gated-continual-cartridges-data) | ~5 GB | phase corpora, eval sets, self-study **synthetic training data** (per benchmark) | `data/` |
| [`qtris123/gated-continual-cartridges-caches`](https://huggingface.co/datasets/qtris123/gated-continual-cartridges-caches) | ~132 GB | compacted **KV caches** (qasper p01–p04) + 5-phase continual-compaction run trees (finqa/quality/techqa) | `outputs/caches/`, `outputs/*_5phase_runs/` |
| [`qtris123/gated-continual-cartridges-results`](https://huggingface.co/datasets/qtris123/gated-continual-cartridges-results) | ~1 GB | evaluations, 5-phase perplexity metrics, sweep experiments, **plots/tables/figures**, recipes | `outputs/` |

Everything is stored **machine-independently**: paths in shards/indices are
relative, symlinks are recorded and rehydrated on extraction, and the cache
index root is rewritten to your local path during setup.

## Requirements

- `hf` CLI (`pip install huggingface_hub`) — the project `.venv` already has it.
- `zstd` (`apt install zstd`).
- `HF_TOKEN` in the environment (needed for private repos and for uploading):
  ```bash
  export HF_TOKEN=hf_...
  ```

## Setup / prepare (download)

From the repo root, run the one entry point:

```bash
# everything (data + results + caches)  — caches is ~132 GB, downloaded last
scripts/prepare_artifacts.sh --which all

# or fetch only what you need
scripts/prepare_artifacts.sh --which data       # datasets + synth
scripts/prepare_artifacts.sh --which results    # evals, plots, tables
scripts/prepare_artifacts.sh --which caches     # the big KV caches
```

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--which data\|caches\|results\|all` | `all` | what to fetch |
| `--dest <dir>` | repo root | where to populate `data/` and `outputs/` |
| `--user <hf-user>` | `qtris123` | HF namespace |
| `--token <hf_token>` | `$HF_TOKEN` | HF token |
| `--keep-archives` | off | keep the downloaded snapshots under `.hf_download/` |

What it does, per target:

- **data** → downloads the tree into `data/`, verifies `MANIFEST.tsv`, checks the
  checkpoint-shard `SHARDS.sha256`, extracts `synth-checkpoints-<ds>.tar.zst` back
  into `data/<ds>/synth/.../checkpoints/`, and rehydrates the `train/` + eval
  symlinks from `SYMLINKS.tsv`.
- **caches** → verifies `SHARDS.sha256`, extracts `caches-qasper-p0X.tar.zst` into
  `outputs/caches/` and `caches-<ds>-5phase-runs.tar.zst` into `outputs/`,
  normalizes absolute symlinks to relative, and writes
  `outputs/caches/index.json` with the local absolute `root`.
- **results** → installs `evaluations/`, `experiments/`, `recipes/`, `eval_plans/`,
  `figures/` under `outputs/`, and `state/<ds>` as `outputs/<ds>_5phase_state/`.

The download is resumable and integrity-checked; re-running is idempotent.

## Layout after prepare

```
data/<dataset>/{phases,synth,train,eval,raw,init_text}/   # inputs + synth data
outputs/caches/<dataset>/<stage>/<technique>/cache.pt     # qasper compacted caches
outputs/caches/index.json                                 # cache index (local root)
outputs/<ds>_5phase_runs/                                  # finqa/quality/techqa run caches
outputs/<ds>_5phase_state/                                 # per-phase perplexity metrics
outputs/{evaluations,experiments,recipes,eval_plans,figures}/
```

## Rebuild & re-upload (maintainers)

Staging trees are built locally (relative paths, tar shards, manifests, cards),
then the datasets are deleted + recreated + uploaded.

```bash
# 1) build machine-independent staging under .hf_staging/
bash scripts/hf/build_data.sh
bash scripts/hf/build_caches.sh       # slow: tars ~159 GB of caches
bash scripts/hf/build_results.sh
.venv/bin/python scripts/hf/make_cards.py .hf_staging   # dataset cards (README.md)

# 2) delete + recreate + upload (requires HF_TOKEN with write access)
scripts/hf/upload_all.sh data
scripts/hf/upload_all.sh results
scripts/hf/upload_all.sh caches
# or: scripts/hf/upload_all.sh all
```

Files:

- `config.sh` — repo IDs, local paths, packaging knobs (source'd by the others).
- `build_data.sh` — browsable `data/` + `checkpoint-shards/` + `SYMLINKS.tsv` + manifests.
- `build_caches.sh` — per-stage/per-benchmark `zstd` tar shards + relative `index.json`.
- `build_results.sh` — evals/experiments/state/figures staging.
- `upload_all.sh` — `delete_repo` → `create_repo` → `hf upload-large-folder`.
- `relativize_index.py` — rewrite `outputs/caches/index.json` to relative paths.
- `make_cards.py` — generate the per-dataset `README.md` cards.

> Notes
> - `.hf_staging/` and `.hf_download/` are git-ignored (local scratch only).
> - Uploads benefit from HuggingFace **xet** chunk-dedup (large caches transfer
>   far fewer bytes than their on-disk size).
> - Some `dataset_clean.parquet` files referenced by `data/CORRUPTION_AUDIT.json`
>   are regenerated locally rather than shipped.
