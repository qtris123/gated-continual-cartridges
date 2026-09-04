"""Phase 1: Train initial qasper cartridge (no sparse / no bg_stats).

Standard Phase 1 training without sparse-finetuning machinery — use this when you
do NOT plan to run sparse Phase 2 with TF-IDF later. For sparse Phase 2 prep, use
`train/initial_sparse.py` instead, which collects background statistics.

Set ``BASELINE=1`` to reproduce the published no-cartridge baseline cartridges
(adds the ``baseline`` wandb tag and the 512-token/baseline run-name defaults);
this folds in the old ``baseline_initial.py``.

Usage:
    TEXT_PATH=/path/to/qasper_init_1024.txt \\
    SYNTH_DATA_PATH=/path/to/phase1_QA.parquet \\
    python examples/shared/train/initial_perplexity.py

    # Or with torchrun for multi-GPU:
    TEXT_PATH=... SYNTH_DATA_PATH=... \\
    torchrun --nproc_per_node=2 examples/shared/train/initial_perplexity.py

Env vars:
    TEXT_PATH               (required) — path to source text file
    SYNTH_DATA_PATH         (required) — path to synthesized training data parquet
    EVAL_DATA_PATH          (optional) — path to evaluation data parquet (perplexity)
    NUM_TOKENS              — cartridge size (default: 1024)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 10)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    EVAL_EVERY_N_STEPS      — eval interval in optimizer steps (default: 50)
    SAVE_EVERY_N_STEPS      — save checkpoint every N steps (default: 256)
    DISTRIBUTED_BACKEND     — distributed training backend (default: gloo)
    RUN_NAME                — W&B run name (default: qasper_phase1)
"""

import os

import pydrantic

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.utils.wandb import WandBConfig

BASELINE = os.environ.get("BASELINE", "0") in ("1", "true", "True")
TEXT_PATH = os.environ["TEXT_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "512" if BASELINE else "1024"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "50"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")
RUN_NAME = os.environ.get(
    "RUN_NAME", "qasper_baseline_phase1" if BASELINE else "qasper_phase1"
)
_TAGS = ["train", "qasper", "phase1"] + (["baseline"] if BASELINE else [])

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
    optimizer="adam",
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
    # Perplexity eval (optional) — set EVAL_DATA_PATH to a parquet to enable
    loss_eval_every_n_steps=EVAL_EVERY_N_STEPS if EVAL_DATA_PATH else None,
    loss_evals=(
        [
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=EVAL_DATA_PATH, type="local"),
                    packed_seq_length=2048,
                ),
                name_for_wandb="qasper_perplexity",
            )
        ]
        if EVAL_DATA_PATH
        else []
    ),
    save_every_n_steps=SAVE_EVERY_N_STEPS,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(tags=_TAGS),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
