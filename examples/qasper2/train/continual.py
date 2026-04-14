"""Phase 2: Continual update of a cartridge on new synthetic data.

Loads a Phase 1 cartridge from a local checkpoint and continues training
on new synthetic data. Evaluates on both phase 1 (forgetting detection)
and phase 2 eval sets.

Usage:
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \
    SYNTH_DATA_PATH=/path/to/dataset_phase2.parquet \
    EVAL_DATA_PATH=/path/to/eval.parquet \
    python examples/qasper2/train/continual.py

Env vars:
    PHASE1_CACHE_PATH  (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH    (required) — path to synthesized training data parquet
    EVAL_DATA_PATH     (required) — path to evaluation data parquet
    NUM_TOKENS         — cartridge size (default: -1; used in run name only)
    MODEL_NAME         — HuggingFace model ID (default: meta-llama/Llama-3.2-3B-Instruct; use Qwen/Qwen3-* for Qwen)
    COMPANY            — company name for run naming (default: unknown)
    YEAR_Y             — year for run naming (default: unknown)
    PROVENANCE_TAG     — tag appended to last user message per convo (default: no tag)
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.cache import TrainableCache, KVCacheFactory
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.utils.wandb import WandBConfig


class KVFromLocal(KVCacheFactory):
    """Initialize a KV cache from a local checkpoint file."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")



PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ["EVAL_DATA_PATH"]
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "-1"))  # used in run name only; cache size is set by PHASE1_CACHE_PATH
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "128"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")

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
            ),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    loss_eval_every_n_steps= EVAL_EVERY_N_STEPS,
    loss_evals=[
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(
                    path=EVAL_DATA_PATH,
                    type="local",
                ),
                packed_seq_length=2048,
            ),
            name_for_wandb="qasper_perplexity",
        )
    ],
    save_every_n_steps=SAVE_EVERY_N_STEPS,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(tags=["train", "qasper", "phase2"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        f"qasper_phase2_{MODEL_NAME.split('/')[-1]}_{NUM_TOKENS}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)