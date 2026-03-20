"""Phase 1: Train initial cartridge on synthetic data.

Usage:
    SYNTH_DATA_PATH=/path/to/dataset_clean.parquet \
    python experiments/experiments_2/train_initial.py

    # Or with torchrun for multi-GPU:
    SYNTH_DATA_PATH=/path/to/dataset_clean.parquet \
    torchrun --nproc_per_node=2 experiments/experiments_2/train_initial.py

Env vars:
    SYNTH_DATA_PATH  (required) — path to synthesized training data parquet
    TEXT_PATH        — path to source text (default: data/texts/AMD_2022_10K.txt)
    EVAL_PHASE1_PATH — path to phase 1 eval parquet (default: data/eval/eval_phase1.parquet)
    NUM_TOKENS       — cartridge size (default: 4096)
    MODEL_NAME       — HuggingFace model ID (default: meta-llama/Llama-3.2-3B-Instruct; use Qwen/Qwen3-* for Qwen)
    COMPANY          — company name for run naming (default: unknown)
    YEAR_Y           — year for run naming (default: unknown)
    PROVENANCE_TAG   — tag appended to last user message per convo, e.g. "AMD 2022" (default: no tag)
"""

import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig
from cartridges.utils.wandb import WandBConfig

SCRIPT_DIR = Path(__file__).resolve().parent

SYNTH_DATA_PATH = os.environ.get("SYNTH_DATA_PATH")
TEXT_PATH = os.environ.get("TEXT_PATH", str(SCRIPT_DIR / "data" / "texts" / "AMD_2021_10K.txt"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
COMPANY = os.environ.get("COMPANY", "unknown")
YEAR_Y  = os.environ.get("YEAR_Y",  "unknown")
PROVENANCE_TAG = os.environ.get("PROVENANCE_TAG", "")

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
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
    wandb=WandBConfig(tags=["train", "experimen_2", "phase1"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        f"{COMPANY}_{YEAR_Y}_phase1_lr{{lr}}_toks{NUM_TOKENS}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)
