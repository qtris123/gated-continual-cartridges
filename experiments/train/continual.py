"""Phase 2: Continual update of a cartridge on new synthetic data.

Loads a Phase 1 cartridge from a local checkpoint and continues training
on new synthetic data. Evaluates on both phase 1 (forgetting detection)
and phase 2 eval sets.

Usage:
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \
    SYNTH_DATA_PATH=/path/to/dataset_phase2.parquet \
    python experiments/train/continual.py

Env vars:
    PHASE1_CACHE_PATH  (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH    (required) — path to synthesized training data parquet
    NUM_TOKENS         — cartridge size (default: 1024)
    MODEL_NAME         — HuggingFace model ID (default: meta-llama/Llama-3.2-3B-Instruct; use Qwen/Qwen3-* for Qwen)
    COMPANY            — company name for run naming (default: unknown)
    YEAR_Y             — year for run naming (default: unknown)
    PROVENANCE_TAG     — tag appended to last user message per convo (default: no tag)
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset, TrainDataset
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig, TrainConfig
from cartridges.utils.wandb import WandBConfig


class KVFromLocal(KVCacheFactory):
    """Initialize a KV cache from a local checkpoint file."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))  # used in run name only; cache size is set by PHASE1_CACHE_PATH
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
PROVENANCE_TAG = os.environ.get("PROVENANCE_TAG", "")
COMPANY = os.environ.get("COMPANY", "unknown")
YEAR_Y  = os.environ.get("YEAR_Y",  "unknown")

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromLocal.Config(
        path=PHASE1_CACHE_PATH,
    ),
    lr=LR,
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(
                path=SYNTH_DATA_PATH, 
                type="local",
                source_tag=PROVENANCE_TAG if PROVENANCE_TAG else None,
            ),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    save_every_n_steps=256,
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["train", "continual_learning", "phase2"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        f"{COMPANY}_{YEAR_Y}_continual_phase2_lr{{lr}}_toks{NUM_TOKENS}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)
