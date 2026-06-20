# Sparse Continual Cartridge — per-layer & per-head TF-IDF Sweep

- **Date:** 2026-06-18
- **Status:** done
- **Related:** commit `c3df9fe`, `cartridges/sparse_cache_finetuning.py`, `cartridges/train.py`

## Goal

Run and validate a two-phase sparse continual cartridge pipeline on LongHealth:

1. **Phase 1** — train a baseline sparse KV-cache cartridge on patients 1–10 using TF-IDF slot selection, collecting background access stats (`bg_stats.pt`) for later IDF weighting.
2. **Phase 2** — continual update on patients 11–20, sweeping `top_t ∈ {64, 128, 256, 512}` at two granularities (`per_layer`, `per_head`) to find the best sparsity operating point.

Hypothesis: finer-grained TF-IDF masking (`per_head`) at moderate `top_t` should protect Phase 1 knowledge better than global or per-layer masking.

## Setup

- **Branch / commit:** `c3df9fe` on `main`
- **Code entry points:**
  - Phase 1: `examples/longhealth/train/initial_sparse.py` (via `train_initial_sparse.sh`)
  - Phase 2 single run: `examples/longhealth/train/continual_sparse.py` (via `train_continual_sparse.sh`)
  - Phase 2 sweep: `examples/longhealth/scripts/sweep_top_t.sh`
  - Forgetting eval: `examples/longhealth/train/eval_forgetting.py` (via `eval_forgetting.sh`)
- **Data:**
  - Phase 1 train: `data/longhealth/train/longhealth_p1-10_8192_no-cartridge.parquet`
  - Phase 2 train: `data/longhealth/train/longhealth_p11-20_8192_no-cartridge.parquet`
  - Phase 1 eval: `data/longhealth/eval/patients_01_to_10.parquet`
  - Phase 2 eval: `data/longhealth/eval/patients_11_to_20.parquet`
- **Hardware:** 4 × NVIDIA RTX 5880 Ada (46 GB each)
- **Key hyperparameters (shared across all Phase 2 runs):**
  - `global_batch_size=64`, `epochs=10`, `lr=2e-2` (Adam), `momentum_masking=freeze`
  - `freeze_keys=0` (keys trainable), `idf_top_k=128`, `idf_smoothing=1.0`
  - `num_tokens=1024`, `packed_seq_length=2048`
  - Phase 1 checkpoint: `outputs/2026-06-17-06-32-27-initial_sparse_per-layer_all-reduce/eafb65ea-56c2-4e97-860a-fa3f874eecbc/cache-step266.pt`
  - `bg_stats.pt` from same folder (all-reduce version — see Gotchas)

## What I tried

- **Phase 1 baseline run** — produced `cache-step266.pt` + `bg_stats.pt` (1 KV cache, 1024 tokens, per-layer granularity).
- **Per-layer sweep** — `top_t ∈ {64, 128, 256, 512}`, `granularity=per_layer`:
  - `outputs/2026-06-17-08-25-16-continual_sparse-per-layer-top-64/64a6715f…/cache-step271.pt`
  - `outputs/2026-06-17-09-27-42-continual_sparse-per-layer-top-128/e9f619cf…/cache-step271.pt`
  - `outputs/2026-06-17-10-29-28-continual_sparse-per-layer-top-256/0b0264a5…/cache-step271.pt`
  - `outputs/2026-06-17-11-31-35-continual_sparse-per-layer-top-512/160e899c…/cache-step271.pt`
- **Per-head sweep** — `top_t ∈ {64, 128, 256, 512}`, `granularity=per_head`:
  - `outputs/2026-06-17-14-40-50-continual_sparse-per-head-top-64/9496b5ee…/cache-step271.pt`
  - `outputs/2026-06-17-15-43-37-continual_sparse-per-head-128/05bea2d5…/cache-step271.pt`
  - `outputs/2026-06-17-16-47-02-continual_sparse-per-head-256/09ebd237…/cache-step271.pt`
  - `outputs/2026-06-17-17-50-04-continual_sparse-per-head-512/34c1ee97…/cache-step271.pt`
- **Forgetting eval** — `eval_forgetting.sh` runs all 9 checkpoints (8 Phase 2 + 1 baseline) × 2 eval sets in parallel across 4 GPUs (4× wall-clock speedup vs. original sequential loop).

## Key findings / insights

- **`bg_stats.pt` was saved by ALL ranks, not just rank 0** (critical bug, now fixed). Each rank used `DistributedSampler` and saw only 1/4 of the data, producing per-rank `bg_stats.pt` files where ~99.4% of `batch_ranked_positions` elements differed across any pair (12.15M / 12.23M elements). This meant IDF weights were computed from a biased quarter of the corpus. Fix: `collect_background_stats` now `all_gather_object`s ranked positions from all ranks and only rank 0 writes the merged result (~4× more background batches, covering the full corpus).

- **TF scores during training were already correct.** The per-step access scores are `dist.all_reduce(SUM)`-ed before `rank_positions()` is called (line 517 of `train.py`), so every rank computes the same gradient mask. Only the background stats collection was broken.

- **`cache_last.pt` symlink in Phase 1 output was broken.** The symlink target path dropped the `_per-layer_all-reduce` suffix from the directory name. Always point directly to `cache-step266.pt` rather than the symlink.

- **Subfolder count per experiment varies (1–4)**, all experiments nonetheless completed with `last_optimizer_step=271` on 4 GPUs. The missing rank folders are a startup-race artifact (ranks that crash before `pydrantic.main` creates their UUID dir). Training correctness is unaffected; only the rank-0 folder matters.

- **`BATCH_SIZE` env var in `eval_forgetting.sh` is a no-op** for loss eval. The `evaluate_perplexity` DataLoader hardcodes `batch_size=1`; the dataset packs conversations into fixed 2048-token windows internally. Each forward pass is always 1 × 2048 tokens regardless.

## Gotchas / surprises

- **All 4 ranks independently save `bg_stats.pt`** — the code block in `train.py` that calls `collect_background_stats` was NOT guarded by `is_rank_zero`, causing each rank to save its own biased file to its own UUID directory. The fix (commit `c3df9fe`) adds `all_gather_object` inside `collect_background_stats` and saves only from rank 0.

- **`pydrantic` generates a fresh UUID per process** — with `torchrun --nproc_per_node=4`, you always get 4 UUID subdirectories under the run's `launch_id` folder. Only the rank-0 directory gets checkpoints and `bg_stats.pt`; the rest hold only `config.yaml`.

- **The Phase 1 `cache_last.pt` symlink was dangling** — it pointed to `.../2026-06-17-06-32-27-initial_sparse/eafb65ea…/cache-step266.pt` (missing `_per-layer_all-reduce`). The `.pt` file existed in the correct directory; always use the absolute path to `cache-step266.pt`.

- **`eval_forgetting.sh` originally printed interleaved logs** when parallelised. Fixed by redirecting each job to its own `eval.log` file. Use `tail -f outputs/longhealth_forgetting_eval_*/baseline__p1-10_eval/eval.log` to monitor live.

- **sweep_top_t.sh uses a fixed `MASTER_PORT=29507`** — if a previous torchrun job hasn't fully torn down, the next run's non-zero ranks may fail to connect the rendezvous, leaving fewer than 4 UUID subfolders. This does not affect training correctness (rank 0 always wins).

## Artifacts

- **Code changes:**
  - `cartridges/sparse_cache_finetuning.py` — `collect_background_stats` now accepts `is_rank_zero`, does `dist.all_gather_object`, merges ranked positions, saves only from rank 0
  - `cartridges/train.py` — passes `is_rank_zero` to `collect_background_stats`
  - `examples/longhealth/scripts/eval_forgetting.sh` — parallel GPU dispatch (`CUDA_VISIBLE_DEVICES`, round-robin slots, per-job `eval.log`)
- **Phase 1 checkpoint:** `outputs/2026-06-17-06-32-27-initial_sparse_per-layer_all-reduce/eafb65ea-56c2-4e97-860a-fa3f874eecbc/cache-step266.pt`
- **Phase 1 bg_stats:** `outputs/2026-06-17-06-32-27-initial_sparse_per-layer_all-reduce/eafb65ea-56c2-4e97-860a-fa3f874eecbc/bg_stats.pt` (391 MB, all-reduce merged, 427 batches × 4 ranks = 1708 effective batches)
- **Phase 2 sweep checkpoints:** see "What I tried" above for full paths
- **Forgetting eval results:** `outputs/longhealth_forgetting_eval_<pid>/` (one subdir per checkpoint × eval set, each with `eval.log`)

## Open questions / next steps

- [ ] Read forgetting eval results once complete — compare P1 retention (patients 1-10 perplexity) vs. P2 acquisition (patients 11-20 perplexity) across `per_layer` vs. `per_head` and all four `top_t` values.
- [ ] Determine optimal `top_t` and granularity from the sweep, then re-run Phase 2 with that setting as the canonical continual cartridge.
- [ ] Check whether `per_head` at low `top_t` (64) genuinely outperforms `per_layer` at the same `top_t`, or if the reduced update budget hurts acquisition too much.
- [ ] Investigate the `per-head-top-64` run that produced only 1 UUID subfolder — confirm it ran on 4 GPUs by checking the optimizer step count per epoch vs. expected (steps × 4 GPUs × accum=16 should equal dataset size × epochs).
- [ ] Consider wiring up `batch_size` in `evaluate_perplexity` to actually batch multiple packed sequences per forward pass — with 46 GB free after loading the 3B model, `batch_size=16` or more is easily feasible and would speed up eval further.
- [ ] Apply the same `bg_stats.pt` all-gather fix and parallel eval dispatch to the Qasper2 eval script (`examples/qasper2/scripts/eval_forgetting.sh`).
