"""Standalone loss-perplexity eval on a saved cartridge checkpoint.

Loads a local .pt checkpoint and evaluates it on any eval parquet — use
qasper_eval_QA.parquet to measure Phase 1 forgetting after Phase 2 training.

Usage:
    CHECKPOINT_PATH=/path/to/cache_last.pt \
    EVAL_DATA_PATH=/home/vo43/cartridges/examples/qasper2/qasper_eval_QA.parquet \
    python examples/qasper2/train/eval_forgetting.py

    # Multi-GPU:
    CHECKPOINT_PATH=/path/to/cache_last.pt \
    EVAL_DATA_PATH=.../qasper_eval_QA.parquet \
    torchrun --nproc_per_node=2 examples/qasper2/train/eval_forgetting.py

Env vars:
    CHECKPOINT_PATH     (required) — local path to a .pt cartridge checkpoint
    EVAL_DATA_PATH      (required) — eval parquet (QA = Phase 1, MT = Phase 2)
    MODEL_NAME          — HF model id (default: meta-llama/Llama-3.2-3B-Instruct)
    RUN_NAME            — label shown in wandb (default: forgetting_eval)
    BATCH_SIZE          — eval batch size (default: 4)
    WANDB_DISABLED      — set to 1 to skip wandb logging
"""

import os

import pydrantic

from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.evaluate import LossEvalRunConfig
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig
from cartridges.utils.wandb import WandBConfig


class KVFromLocal(KVCacheFactory):
    """Load a KV cache from a local checkpoint file."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


CHECKPOINT_PATH = os.environ["CHECKPOINT_PATH"]
EVAL_DATA_PATH = os.environ["EVAL_DATA_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
RUN_NAME = os.environ.get("RUN_NAME", "forgetting_eval")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "0") in ("1", "true", "True")

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

config = LossEvalRunConfig(
    name=RUN_NAME,
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromLocal.Config(
        path=CHECKPOINT_PATH,
    ),
    eval=LossEvalConfig(
        dataset=LossEvalDataset.Config(
            data_source=DataSource(
                path=EVAL_DATA_PATH,
                type="local",
            ),
            packed_seq_length=2048,
        ),
        name_for_wandb="qasper_perplexity",
    ),
    batch_size=BATCH_SIZE,
    wandb=None if WANDB_DISABLED else WandBConfig(
        tags=["eval", "forgetting", RUN_NAME],
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)
