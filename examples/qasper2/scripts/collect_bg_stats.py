"""Collect background access stats (IDF) for a Phase-1 cartridge at a chosen granularity.

Needed because per_head TF-IDF requires per_head background stats; the Phase-1
run only saved per_layer stats.

Usage:
  PHASE1_CACHE_PATH=.../cache_last.pt \\
  QA_DATA_PATH=data/qasper/train/qwen_qasper_QA_task_8192.parquet \\
  GRANULARITY=per_head \\
  OUT_PATH=.../bg_stats_per_head.pt \\
  python examples/qasper2/scripts/collect_bg_stats.py
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.datasets import DataSource, TrainDataset
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import (
    SparseCacheFinetuningConfig,
    collect_background_stats,
)
from cartridges.train import CacheAndModel
from cartridges.utils import get_logger, seed_everything

logger = get_logger(__name__)

PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
QA_DATA_PATH = os.environ["QA_DATA_PATH"]
GRANULARITY = os.environ.get("GRANULARITY", "per_head")
OUT_PATH = os.environ["OUT_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
NUM_BG_BATCHES = int(os.environ.get("NUM_BG_BATCHES", "1000"))
SEED = int(os.environ.get("SEED", "42"))

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM


def _collate_first(batch):
    return batch[0]


def main():
    seed_everything(SEED)
    local_rank = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = (
        HFModelConfig(pretrained_model_name_or_path=MODEL_NAME, model_cls=_model_cls)
        .instantiate()
        .to(local_rank)
        .to(torch.bfloat16)
    )
    for p in model.parameters():
        p.requires_grad = False

    cache = TrainableCache.from_pretrained(PHASE1_CACHE_PATH, device="cuda").to(local_rank)

    dataset = TrainDataset.Config(
        data_sources=[DataSource(path=QA_DATA_PATH, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ).instantiate(tokenizer=tokenizer, seed=SEED)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=_collate_first, num_workers=0)

    sparse_cfg = SparseCacheFinetuningConfig(
        enabled=False,
        collect_background_stats=True,
        num_background_batches=NUM_BG_BATCHES,
        granularity=GRANULARITY,
    )
    wrapped = CacheAndModel(cache, model, sparse_config=sparse_cfg)

    t0 = time.time()
    collect_background_stats(
        wrapped_model=wrapped,
        cache=cache,
        dataloader=dataloader,
        config=sparse_cfg,
        local_rank=local_rank,
        save_path=OUT_PATH,
        is_ddp=False,
        is_rank_zero=True,
    )
    logger.info(
        "Collected %s bg_stats in %.1fs → %s",
        GRANULARITY,
        time.time() - t0,
        OUT_PATH,
    )


if __name__ == "__main__":
    main()
