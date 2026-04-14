"""Phase 1: Train initial cartridge.

Usage:
    TEXT_PATH=/path/to/AMD_2021_10K.txt \
    SYNTH_DATA_PATH=/path/to/dataset.parquet \
    EVAL_DATA_PATH=/path/to/eval.parquet \
    python examples/qasper2/train/initial.py

    # Or with torchrun for multi-GPU:
    TEXT_PATH=/path/to/AMD_2021_10K.txt \
    SYNTH_DATA_PATH=/path/to/dataset.parquet \
    EVAL_DATA_PATH=/path/to/eval.parquet \
    torchrun --nproc_per_node=2 examples/qasper2/train/initial.py

Env vars:
    TEXT_PATH               (required) — path to source text file
    SYNTH_DATA_PATH         (required) — path to synthesized training data parquet
    EVAL_DATA_PATH          (required) — path to evaluation data parquet
    NUM_TOKENS              — cartridge size (default: 512)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 1)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    TOP_K_LOGITS            — top-k logits to store (default: 20)
    PACKED_SEQ_LENGTH       — packed sequence length (default: 2048)
    SAVE_EVERY_N_STEPS      — save checkpoint every N steps (default: 256)
    DISTRIBUTED_BACKEND     — distributed training backend (default: gloo)
    COMPANY                 — company name for run naming (default: unknown)
    YEAR_Y                  — year for run naming (default: unknown)
    PROVENANCE_TAG          — tag appended to last user message per convo (default: no tag)
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.utils.wandb import WandBConfig

TEXT_PATH = os.environ["TEXT_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ["EVAL_DATA_PATH"]
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "512"))
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
    kv_cache_initializer=KVFromText.Config(
        text_source=TEXT_PATH,
        max_tokens=NUM_TOKENS,
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
    wandb=WandBConfig(tags=["train", "qasper", "phase1"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        f"qasper_phase1_{MODEL_NAME.split('/')[-1]}_toks{NUM_TOKENS}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)