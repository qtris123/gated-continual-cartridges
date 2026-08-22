"""Phase 1 baseline: dense gradient cartridge training (benchmark reproduction).

Standard gradient-based Phase 1 training without sparse/TF-IDF or AM machinery.
Use this to reproduce the published baseline cartridges on HuggingFace, e.g.:
  - qtris123/llama_qasper-QA-task_8192_512_no-cartridge_10-epochs
  - qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs

For AM or sparse Phase 2 pipelines, use initial_am.py / initial_sparse.py instead.

Usage:
    TEXT_PATH=data/qasper/init_text/qasper_init_512.txt \\
    SYNTH_DATA_PATH=data/qasper/train/qasper_QA_task_8192_no-cartridge.parquet \\
    python examples/shared/train/baseline_initial.py

    # Multi-GPU:
    TEXT_PATH=... SYNTH_DATA_PATH=... \\
    torchrun --nproc_per_node=2 examples/shared/train/baseline_initial.py

Env vars:
    TEXT_PATH               (required) — path to source text file
    SYNTH_DATA_PATH         (required) — path to synthesized training data parquet
    EVAL_DATA_PATH          (optional) — path to evaluation data parquet (perplexity)
    NUM_TOKENS              — cartridge size (default: 512)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 10)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    EVAL_EVERY_N_STEPS      — eval interval in optimizer steps (default: 50)
    SAVE_EVERY_N_STEPS      — save checkpoint every N steps (default: 256)
    DISTRIBUTED_BACKEND     — distributed training backend (default: gloo)
    RUN_NAME                — W&B run name (default: qasper_baseline_phase1)
"""

import os

import pydrantic

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.utils.wandb import WandBConfig

TEXT_PATH = os.environ["TEXT_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "512"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "50"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")
RUN_NAME = os.environ.get("RUN_NAME", "qasper_baseline_phase1")

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
    wandb=WandBConfig(tags=["train", "qasper", "phase1", "baseline"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
