"""Phase 2: Continual update of a qasper cartridge (no sparse / no TF-IDF).

Loads a Phase 1 cartridge from a local checkpoint and continues training on
new synthetic data (e.g. MT task). This variant performs *dense* finetuning of
the cache — every position is updated. For sparse TF-IDF Phase 2, use
`train/continual_sparse.py` instead.

Usage:
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \\
    SYNTH_DATA_PATH=/path/to/phase2_MT.parquet \\
    python examples/qasper2/train/continual.py

    # Multi-GPU:
    PHASE1_CACHE_PATH=... SYNTH_DATA_PATH=... \\
    torchrun --nproc_per_node=2 examples/qasper2/train/continual.py

Env vars:
    PHASE1_CACHE_PATH       (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH         (required) — path to synthesized training data parquet
    EVAL_DATA_PATH          (optional) — path to evaluation data parquet (perplexity)
    NUM_TOKENS              — cartridge size (default: -1; used in run name only)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 10)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    EVAL_EVERY_N_STEPS      — eval every N optimizer steps (default: 15)
    SAVE_EVERY_N_STEPS      — save checkpoint every N steps (default: 256)
    DISTRIBUTED_BACKEND     — distributed training backend (default: gloo)
    RUN_NAME                — W&B run name (default: qasper_phase2)
"""

import os

import pydrantic

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.cache import TrainableCache, KVCacheFactory
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig, LossEvalConfig, CosWithWarmup
from cartridges.utils.wandb import WandBConfig


class KVFromLocal(KVCacheFactory):
    """Initialize a KV cache from a local checkpoint file."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "-1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "15"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")
RUN_NAME = os.environ.get("RUN_NAME", "qasper_phase2")

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromLocal.Config(
        path=PHASE1_CACHE_PATH,
    ),
    optimizer="adam",
    lr=LR,
    lr_scheduler=CosWithWarmup.Config(
        max_steps=500,
        warmup_steps=20,
        alpha_f=0.1,
    ),
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
    # Perplexity eval (optional) — set EVAL_DATA_PATH to a parquet to enable.
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
    wandb=WandBConfig(tags=["train", "qasper", "phase2"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
