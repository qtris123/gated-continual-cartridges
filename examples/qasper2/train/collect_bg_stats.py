"""SETUP-BG1: standalone collector for IDF background statistics.

Runs forward-only passes over a BACKGROUND corpus against a loaded Phase-1
cartridge and saves a ``BackgroundAccessTracker`` (``bg_stats.pt``) so that
future ``USE_IDF=1`` AM-sparse Phase-2 runs can load it via
``examples/qasper2/train/continual_am_sparse.py``'s ``bg_tracker.load()`` path.

This is INFRASTRUCTURE (no hypothesis). Single-process, GPU forward-only,
``is_ddp=False`` (avoids the all-gather merge path).

Construction mirrors the working patterns in
``examples/qasper2/train/continual_am_sparse.py``:
  * model  = HFModelConfig(...).instantiate().to(local_rank).to(bf16), params frozen
  * cache  = TrainableCache.from_pretrained(PHASE1_CACHE_PATH, device="cuda")
  * dataset= TrainDataset.Config(data_sources=[DataSource(path=..., type="local")],
             top_k_logits=20, packed_seq_length=2048, packing_mode="truncate")
  * wrapped= CacheAndModel(cache, model)   # no am/sparse config -> collect
             installs its own query-capture hooks

Env (all optional; defaults target SETUP-BG1):
  PHASE1_CACHE_PATH  outputs/phase1_selfdistill_qwen512/cache_last.pt
  BG_CORPUS_PATH     data/qasper/train/qwen_qasper_QA_task_8192.parquet
  BG_STATS_OUT       outputs/phase1_selfdistill_qwen512/bg_stats.pt
  GRANULARITY        per_layer
  MODEL_NAME         Qwen/Qwen3-4B-Instruct-2507
  NUM_BG_BATCHES     999999999   (map-style DataLoader stops after one epoch)
  SEED               42
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.datasets import DataSource, TrainDataset
from cartridges.sparse_cache_finetuning import (
    BackgroundAccessTracker,
    CacheTFIDFRanker,
    SparseCacheFinetuningConfig,
    collect_background_stats,
)
from cartridges.train import CacheAndModel

PHASE1_CACHE_PATH = os.environ.get(
    "PHASE1_CACHE_PATH", "outputs/phase1_selfdistill_qwen512/cache_last.pt"
)
BG_CORPUS_PATH = os.environ.get(
    "BG_CORPUS_PATH", "data/qasper/train/qwen_qasper_QA_task_8192.parquet"
)
BG_STATS_OUT = os.environ.get(
    "BG_STATS_OUT", "outputs/phase1_selfdistill_qwen512/bg_stats.pt"
)
GRANULARITY = os.environ.get("GRANULARITY", "per_layer")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
NUM_BG_BATCHES = int(os.environ.get("NUM_BG_BATCHES", "999999999"))
IDF_TOP_K = int(os.environ.get("IDF_TOP_K", "128"))
SEED = int(os.environ.get("SEED", "42"))

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM


def _collate_first(batch):
    return batch[0]


def main():
    t_start = time.time()
    local_rank = "cuda" if torch.cuda.is_available() else "cpu"
    assert local_rank == "cuda", "SETUP-BG1 requires a CUDA device."

    print(f"[SETUP-BG1] cache={PHASE1_CACHE_PATH}", flush=True)
    print(f"[SETUP-BG1] bg_corpus={BG_CORPUS_PATH}", flush=True)
    print(f"[SETUP-BG1] out={BG_STATS_OUT}", flush=True)
    print(f"[SETUP-BG1] granularity={GRANULARITY} model={MODEL_NAME} "
          f"num_bg_batches={NUM_BG_BATCHES}", flush=True)

    # --- model (frozen, bf16) ---
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = (
        HFModelConfig(pretrained_model_name_or_path=MODEL_NAME, model_cls=_model_cls)
        .instantiate()
        .to(local_rank)
        .to(torch.bfloat16)
    )
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Phase-1 cartridge ---
    # from_pretrained maps loaded weights via map_location, but the internally
    # constructed _seq_ids / _init_seq_ids buffers stay on CPU; the extra
    # .to(local_rank) moves them to the device (mirrors continual_am_sparse.py's
    # _init_model_and_cache, which .to(local_rank)'s the cache).
    cache = TrainableCache.from_pretrained(PHASE1_CACHE_PATH, device="cuda").to(local_rank)
    n_trainable = cache._num_trainable_tokens
    n_layers = len(cache.trainable_keys)
    n_kv_heads = cache.trainable_keys[0].shape[1]
    print(f"[SETUP-BG1] cache: n_trainable={n_trainable} n_layers={n_layers} "
          f"n_kv_heads={n_kv_heads} head_dim={cache.config.head_dim}", flush=True)

    # --- background corpus (same TrainDataset.Config as continual_am_sparse.py) ---
    ds = TrainDataset.Config(
        data_sources=[DataSource(path=BG_CORPUS_PATH, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ).instantiate(tokenizer=tok, seed=SEED)
    dl = DataLoader(ds, batch_size=1, collate_fn=_collate_first, num_workers=0)
    print(f"[SETUP-BG1] dataset packed-sequences (len)={len(ds)}", flush=True)

    # enabled=False -> collect_background_stats installs its own query-capture hooks.
    cfg = SparseCacheFinetuningConfig(
        enabled=False,
        collect_background_stats=True,
        num_background_batches=NUM_BG_BATCHES,
        granularity=GRANULARITY,
        background_top_k_per_batch=IDF_TOP_K,
    )

    # No am/sparse config passed -> _am_enabled/_sparse_enabled False -> collect
    # installs its own hooks (had_hooks path).
    wrapped = CacheAndModel(cache, model)

    setup_s = time.time() - t_start
    print(f"[SETUP-BG1] MODEL_LOAD_S={setup_s:.1f}", flush=True)

    # --- collect (GPU forward-only) ---
    t_collect = time.time()
    collect_background_stats(
        wrapped_model=wrapped,
        cache=cache,
        dataloader=dl,
        config=cfg,
        local_rank=local_rank,
        save_path=BG_STATS_OUT,
        is_ddp=False,
        is_rank_zero=True,
    )
    collect_s = time.time() - t_collect
    print(f"[SETUP-BG1] COLLECT_WALLCLOCK_S={collect_s:.1f}", flush=True)

    # --- sanity check: mirror the Phase-2 consumption path exactly ---
    print("[SETUP-BG1] --- SANITY CHECK ---", flush=True)
    raw = torch.load(BG_STATS_OUT, weights_only=False)
    print(f"[SETUP-BG1] SANITY raw_keys={sorted(raw.keys())}", flush=True)
    brp = raw["batch_ranked_positions"]
    n_batches_seen = int(raw["num_batches"])
    print(f"[SETUP-BG1] SANITY shape={tuple(brp.shape)} dtype={brp.dtype} "
          f"num_batches={n_batches_seen} granularity={raw['granularity']}", flush=True)

    # (a) loads (b) per-layer stats span all layers (c) values finite/valid
    assert raw["granularity"] == GRANULARITY, "granularity mismatch in saved file"
    assert brp.dim() == 3, f"expected (n_batches, n_layers, n_tokens), got {tuple(brp.shape)}"
    assert brp.shape[1] == n_layers, f"layer dim {brp.shape[1]} != cache n_layers {n_layers}"
    assert brp.shape[2] == n_trainable, f"token dim {brp.shape[2]} != n_trainable {n_trainable}"
    assert int(brp.min()) >= 0 and int(brp.max()) < n_trainable, "ranked position indices out of range"

    # Reconstruct via the EXACT Phase-2 load path (continual_am_sparse.py).
    bg_tracker = BackgroundAccessTracker(num_batches=n_batches_seen, granularity=GRANULARITY)
    bg_tracker.load(BG_STATS_OUT)
    assert bg_tracker.current_batch_id == n_batches_seen

    # Build the ranker exactly as continual_am_sparse.py does and check finite IDF.
    ranker = CacheTFIDFRanker(
        background_tracker=bg_tracker,
        use_idf=True,
        smoothing=1.0,
        top_k_per_batch=IDF_TOP_K,
        granularity=GRANULARITY,
    )
    idf = ranker.get_idf_scores()
    df = ranker.get_document_frequencies()
    assert idf is not None and df is not None, "ranker produced no IDF/DF"
    assert tuple(idf.shape) == (n_layers, n_trainable), f"idf shape {tuple(idf.shape)}"
    assert torch.isfinite(idf).all(), "IDF has non-finite values"
    assert int(df.min()) >= 0 and int(df.max()) <= n_batches_seen, "DF out of range"
    print(f"[SETUP-BG1] SANITY idf_shape={tuple(idf.shape)} "
          f"idf[min/mean/max]={float(idf.min()):.4f}/{float(idf.mean()):.4f}/{float(idf.max()):.4f} "
          f"df[min/max]={int(df.min())}/{int(df.max())} finite=True", flush=True)
    print("[SETUP-BG1] SANITY_OK=1 (Phase-2 bg_tracker.load + CacheTFIDFRanker accept the file)",
          flush=True)

    total_s = time.time() - t_start
    print(f"[SETUP-BG1] RESULT collect_wallclock_s={collect_s:.1f} "
          f"n_layers={n_layers} n_kv_heads={n_kv_heads} "
          f"num_background_batches_seen={n_batches_seen} granularity={GRANULARITY} "
          f"total_s={total_s:.1f}", flush=True)
    print("[SETUP-BG1] DONE", flush=True)


if __name__ == "__main__":
    main()
