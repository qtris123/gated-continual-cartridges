"""Train AMD 10-K cartridges at multiple sizes.

Usage:
    # Single size:
    NUM_TOKENS=512 python experiments/multi_documents/train_amd.py

    # Or with torchrun for multi-GPU:
    NUM_TOKENS=2048 torchrun --nproc_per_node=2 experiments/multi_documents/train_amd.py

Cartridge sizes: set NUM_TOKENS to one of {512, 1024, 2048, 4096}.
"""

import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig
from cartridges.train import TrainConfig
from cartridges.utils.wandb import WandBConfig

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_PATH = str(SCRIPT_DIR / "data" / "texts" / "AMD_2022_10K.txt")

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "16384"))

# Point to the synthesized dataset. By default, look for a local parquet file
# in CARTRIDGES_OUTPUT_DIR. Adjust path to match your synthesis output.
SYNTH_DATA_PATH =  "/home/phudishp/cartridges/outputs/2026-02-09-15-23-19-synthesize_amd/synthesize_amd_meta-llama/Llama-3.2-3B-Instruct_n65536-0/artifact/dataset_clean.parquet"

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
    save_every_n_steps=256,
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["train", "multi_doc", "amd"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        "amd_train_lr{lr}_toks{kv_cache_initializer.max_tokens}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)
