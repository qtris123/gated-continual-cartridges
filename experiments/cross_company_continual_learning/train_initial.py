"""Phase 1: Train initial cartridge on AMD 2022 10-K.

Usage:
    SYNTH_DATA_PATH=/path/to/amd_dataset_clean.parquet \
    python experiments/sequential_documents/train_initial.py

    # Or with torchrun for multi-GPU:
    SYNTH_DATA_PATH=/path/to/amd_dataset_clean.parquet \
    torchrun --nproc_per_node=2 experiments/sequential_documents/train_initial.py

Env vars:
    SYNTH_DATA_PATH  (required) — path to AMD synthesized training data parquet
    TEXT_PATH         — path to source text (default: data/texts/AMD_2022_10K.txt)
    EVAL_DATA_PATH    — path to AMD eval parquet (default: data/eval/eval_amd.parquet)
    NUM_TOKENS        — cartridge size (default: 1024)
"""

import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, LossEvalDataset, TrainDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig, TrainConfig
from cartridges.utils.wandb import WandBConfig

SCRIPT_DIR = Path(__file__).resolve().parent

TEXT_PATH = os.environ.get(
    "TEXT_PATH", str(SCRIPT_DIR / "data" / "texts" / "AMD_2022_10K.txt")
)
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ.get(
    "EVAL_DATA_PATH", str(SCRIPT_DIR / "data" / "eval" / "eval_amd.parquet")
)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=TEXT_PATH,
        max_tokens=NUM_TOKENS,
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
                data_source=DataSource(path=EVAL_DATA_PATH, type="local"),
                packed_seq_length=2048,
            ),
            name_for_wandb="amd_eval",
        ),
    ],
    save_every_n_steps=256,
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["train", "sequential_documents", "phase1"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        "sequential_phase1_lr{lr}_toks{kv_cache_initializer.max_tokens}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)
