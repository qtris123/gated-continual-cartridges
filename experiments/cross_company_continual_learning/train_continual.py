"""Phase 2: Continue training cartridge with PepsiCo 2022 10-K.

Loads the Phase 1 cartridge (AMD 2022) from a local checkpoint and continues
training on PepsiCo synthetic data. Evaluates on AMD (forgetting), PepsiCo
(learning), and cross-document (cross-document reasoning) eval sets.

Usage:
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \
    SYNTH_DATA_PATH=/path/to/pepsi_dataset_clean.parquet \
    python experiments/sequential_documents/train_continual.py

Env vars:
    PHASE1_CACHE_PATH  (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH    (required) — path to PepsiCo synthesized training data parquet
    EVAL_AMD_PATH      — path to AMD eval parquet (default: data/eval/eval_amd.parquet)
    EVAL_PEPSI_PATH    — path to PepsiCo eval parquet (default: data/eval/eval_pepsi.parquet)
    EVAL_CROSS_DOC_PATH — path to cross-doc eval parquet (default: data/eval/eval_cross_doc.parquet)
    NUM_TOKENS         — cartridge size (default: 16384)
"""

import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset, TrainDataset
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig, TrainConfig
from cartridges.utils.wandb import WandBConfig


class KVFromLocal(KVCacheFactory):
    """Initialize a KV cache from a local checkpoint file."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


SCRIPT_DIR = Path(__file__).resolve().parent

PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_AMD_PATH = os.environ.get(
    "EVAL_AMD_PATH", str(SCRIPT_DIR / "data" / "eval" / "eval_amd.parquet")
)
EVAL_PEPSI_PATH = os.environ.get(
    "EVAL_PEPSI_PATH", str(SCRIPT_DIR / "data" / "eval" / "eval_pepsi.parquet")
)
EVAL_CROSS_DOC_PATH = os.environ.get(
    "EVAL_CROSS_DOC_PATH", str(SCRIPT_DIR / "data" / "eval" / "eval_cross_doc.parquet")
)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "16384"))

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    ),
    kv_cache_initializer=KVFromLocal.Config(
        path=PHASE1_CACHE_PATH,
    ),
    lr=2e-2,
    epochs=1,
    global_batch_size=32,
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(path=SYNTH_DATA_PATH, type="local"),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    loss_eval_every_n_steps=1,
    loss_evals=[
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path=EVAL_AMD_PATH, type="local"),
                packed_seq_length=2048,
            ),
            name_for_wandb="amd_eval",
        ),
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path=EVAL_PEPSI_PATH, type="local"),
                packed_seq_length=2048,
            ),
            name_for_wandb="pepsi_eval",
        ),
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path=EVAL_CROSS_DOC_PATH, type="local"),
                packed_seq_length=2048,
            ),
            name_for_wandb="cross_doc_eval",
        ),
    ],
    save_every_n_steps=256,
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["train", "sequential_documents", "phase2"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        f"sequential_phase2_lr{{lr}}_toks{NUM_TOKENS}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)
