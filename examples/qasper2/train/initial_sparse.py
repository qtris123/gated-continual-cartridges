"""Phase 1: Train initial qasper cartridge with background-statistics collection for TF-IDF sparse finetuning.

Extends the standard qasper Phase 1 training by enabling background-stats collection
(collect_background_stats=True) so that Phase 2 can use IDF-weighted sparse cache updates.
The GRANULARITY must match the GRANULARITY used in Phase 2 (train/continual_sparse.py).

Usage:
    TEXT_PATH=/path/to/qasper_init_1024.txt \\
    SYNTH_DATA_PATH=/path/to/phase1_QA.parquet \\
    python examples/qasper2/train/initial_sparse.py

    # Multi-GPU:
    TEXT_PATH=... SYNTH_DATA_PATH=... \\
    torchrun --nproc_per_node=4 examples/qasper2/train/initial_sparse.py

Env vars:
    TEXT_PATH               (required) — path to source text file (qasper context corpus)
    SYNTH_DATA_PATH         (required) — path to synthesized training parquet (Phase 1 task: QA)
    EVAL_DATA_PATH          (optional) — path to evaluation parquet (perplexity in W&B)
    NUM_TOKENS              — cartridge size (default: 1024)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 10)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    EVAL_EVERY_N_STEPS      — eval interval in optimizer steps (default: 50)
    SAVE_EVERY_N_STEPS      — checkpoint interval (default: 256)
    GRANULARITY             — bg_stats granularity: global | per_layer | per_head (default: per_head)
                              Must match the GRANULARITY used in Phase 2.
    RUN_NAME                — W&B run name (default: auto-generated)
    DISTRIBUTED_BACKEND     — distributed backend (default: gloo)
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import SparseCacheFinetuningConfig
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.utils.wandb import WandBConfig

TEXT_PATH = os.environ["TEXT_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
# Optional: provide a parquet to track perplexity in W&B
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "50"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")
# bg_stats granularity — must match Phase 2 GRANULARITY so IDF shapes align
GRANULARITY = os.environ.get("GRANULARITY", "per_head")  # global | per_layer | per_head
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"qasper_phase1_sparse_{MODEL_NAME.split('/')[-1]}_{NUM_TOKENS}_{GRANULARITY}",
)

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
    max_optimizer_steps=MAX_STEPS,
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
    # Sparse finetuning is disabled in Phase 1; background stats are collected so
    # Phase 2 can use IDF-weighted slot selection. The full sorted ranking of all
    # n_tokens positions is stored; the IDF_TOP_K cutoff is applied in Phase 2.
    sparse_cache_finetuning=SparseCacheFinetuningConfig(
        enabled=False,
        use_idf=False,
        collect_background_stats=True,
        num_background_batches=99999999999,
        granularity=GRANULARITY,
    ),
    save_every_n_steps=SAVE_EVERY_N_STEPS,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(
        tags=["train", "qasper", "phase1", "sparse", GRANULARITY.replace("_", "-")]
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
