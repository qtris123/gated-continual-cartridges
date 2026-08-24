"""QuALITY phase-1 self-distilled cartridge (gradient KV, not AM compaction).

Matches the published QASPER p01 recipe:
  KVFromText init (512 tokens, 1 frozen sink) + 10-epoch Adam on
  assistant top-20 logprobs from the self-study parquet.

Usage:
    TEXT_PATH=data/quality/init_text/quality_p1.txt \\
    SYNTH_DATA_PATH=data/quality/train/qwen_quality_p1_task_8192.parquet \\
    python examples/quality/pipelines/train_initial_selfdistill.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pydrantic

from cartridges.datasets import DataSource, LossEvalDataset, TrainDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig, TrainConfig
from cartridges.utils.wandb import WandBConfig
from examples.shared.paths import ROOT

TEXT_PATH = os.environ["TEXT_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH") or None
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "512"))
NUM_FROZEN_TOKENS = int(os.environ.get("NUM_FROZEN_TOKENS", "1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "50"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "nccl")
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"quality_p1_selfdistill_{MODEL_NAME.split('/')[-1]}_{NUM_TOKENS}",
)
WANDB_PROJECT = os.environ.get("CARTRIDGES_WANDB_PROJECT", "SEACrowd")
WANDB_ENTITY = os.environ.get("CARTRIDGES_WANDB_ENTITY", "vqtri-purdue-university")

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM


def ensure_init_text(path: str) -> None:
    """Write concatenated phase-1 stories if the KVFromText source is missing."""
    target = Path(path)
    if target.is_file() and target.stat().st_size > 0:
        return
    from cartridges.data.quality.resources import QuALITYResource

    target.parent.mkdir(parents=True, exist_ok=True)
    text = QuALITYResource(QuALITYResource.Config(phase=1)).to_string()
    target.write_text(text)


ensure_init_text(TEXT_PATH)

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=TEXT_PATH,
        max_tokens=NUM_TOKENS,
        num_frozen_tokens=NUM_FROZEN_TOKENS,
        system_prompt_template="{text}",
    ),
    optimizer="adam",
    lr=LR,
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=SYNTH_DATA_PATH, type="local")],
        targets="logits",
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    loss_eval_every_n_steps=EVAL_EVERY_N_STEPS if EVAL_DATA_PATH else None,
    loss_evals=(
        [
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=EVAL_DATA_PATH, type="local"),
                    packed_seq_length=2048,
                    targets="tokens",
                ),
                name_for_wandb="quality_p01_perplexity",
            )
        ]
        if EVAL_DATA_PATH
        else []
    ),
    save_every_n_steps=SAVE_EVERY_N_STEPS,
    save_after_training=True,
    save_to_wandb=True,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        tags=["train", "quality", "phase1", "selfdistill"],
        notes=(
            "Gradient self-distill of a 512-token QuALITY p01 cartridge from "
            "qwen_quality_p1_task_8192.parquet assistant top-20 logprobs. "
            f"Init text truncated to {NUM_TOKENS} tokens; frozen sink="
            f"{NUM_FROZEN_TOKENS}."
        ),
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", str(ROOT / "outputs")),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
