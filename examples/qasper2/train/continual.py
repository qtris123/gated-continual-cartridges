"""Phase 2: Continual update of a cartridge on new synthetic data with TF-IDF sparse finetuning.

Loads a Phase 1 cartridge from a local checkpoint and continues training
on new synthetic data. Uses sparse gradient masking (SGD + TF-IDF) to
update only the most task-relevant cache positions, protecting Phase 1 knowledge.

Usage:
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \
    SYNTH_DATA_PATH=/path/to/dataset_phase2.parquet \
    EVAL_DATA_PATH=/path/to/eval.parquet \
    python examples/qasper2/train/continual.py

    # With Phase 1 background stats for IDF:
    BG_STATS_PATH=/path/to/phase1/bg_stats.pt \
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \
    SYNTH_DATA_PATH=/path/to/dataset_phase2.parquet \
    EVAL_DATA_PATH=/path/to/eval.parquet \
    python examples/qasper2/train/continual.py

Env vars:
    PHASE1_CACHE_PATH       (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH         (required) — path to synthesized training data parquet
    EVAL_DATA_PATH          (required) — path to evaluation data parquet
    BG_STATS_PATH           (optional) — path to Phase 1 bg_stats.pt for IDF
    NUM_TOKENS              — cartridge size (default: -1; used in run name only)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2.0 for SGD)
    EPOCHS                  — number of epochs (default: 1)
    TOP_T                   — number of cache positions to update per step (default: 500)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    EVAL_EVERY_N_STEPS      — eval every N optimizer steps (default: 128)
    SAVE_EVERY_N_STEPS      — save checkpoint every N steps (default: 256)
    DISTRIBUTED_BACKEND     — distributed training backend (default: gloo)
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.cache import TrainableCache, KVCacheFactory
from cartridges.sparse_cache_finetuning import SparseCacheFinetuningConfig
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
EVAL_DATA_PATH = os.environ["EVAL_DATA_PATH"]
BG_STATS_PATH = os.environ.get("BG_STATS_PATH", "/home/vo43/cartridges/outputs/2026-06-13-11-47-32-initial-per-layer/62cef748-90a5-4914-8ca5-4788dd89d570/cache_last.pt")
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "-1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "128"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")
RUN_NAME = os.environ.get("RUN_NAME", "qasper_phase2")
TOP_T = int(os.environ.get("TOP_T", "512"))
MOMENTUM_MASKING = os.environ.get("MOMENTUM_MASKING", "hard")  # soft | hard | freeze | decouple
FREEZE_KEYS = os.environ.get("FREEZE_KEYS", "1") not in ("0", "false", "False")
GRANULARITY = os.environ.get("GRANULARITY", "global")          # global | per_layer | per_head
IDF_TOP_K = int(os.environ.get("IDF_TOP_K", "128"))            # top-k per bg batch for df computation
IDF_SMOOTHING = float(os.environ.get("IDF_SMOOTHING", "1.0"))  # Laplace smoothing for IDF

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromLocal.Config(
        path=PHASE1_CACHE_PATH,
    ),
    optimizer= "adam",
    lr=LR,
    lr_scheduler=CosWithWarmup.Config(
        max_steps=500,      # Adjust based on your total optimizer steps
        warmup_steps=20,    # Short warmup for continual learning
        alpha_f=0.1,        # Final LR = 0.1 * initial LR
    ),
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(
                path=SYNTH_DATA_PATH,
                type="local",
                #limit = 1000,
            ),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    sparse_cache_finetuning=SparseCacheFinetuningConfig(
        enabled=True,
        top_t=TOP_T,
        use_idf=BG_STATS_PATH is not None,
        background_indices_path=BG_STATS_PATH,
        collect_background_stats=True,
        num_background_batches=999999999,
        momentum_masking=MOMENTUM_MASKING,
        freeze_keys=FREEZE_KEYS,
        granularity=GRANULARITY,
        background_top_k_per_batch=IDF_TOP_K,
        idf_smoothing=IDF_SMOOTHING,
    ),
    loss_eval_every_n_steps=EVAL_EVERY_N_STEPS,
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
    # !!!
    wandb=WandBConfig(tags=["train", "qasper", "phase2", f"top-t-{TOP_T}", "lr-2e-2", "all-layers", "value-only", "8k-data", "epochs-10", f"momentum-{MOMENTUM_MASKING}", "adam", f"{GRANULARITY}", f"idf-top-k-{IDF_TOP_K}"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)