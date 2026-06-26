"""Phase 2: Continual update of a qasper cartridge with TF-IDF sparse cache finetuning.

Loads a Phase 1 cartridge (trained via initial_sparse.py) and continues training
on Phase 2 data (e.g. MT task). Uses sparse gradient masking (TF-IDF) to update
only the most task-relevant cache positions, protecting Phase 1 knowledge.

The BG_STATS_PATH should point to the cache_last.pt from the Phase 1 run produced
by train/initial_sparse.py — it contains the background position statistics needed
to compute IDF-weighted slot selection.

Usage:
    PHASE1_CACHE_PATH=/path/to/phase1/cache_last.pt \\
    SYNTH_DATA_PATH=/path/to/phase2_MT.parquet \\
    python examples/qasper2/train/continual_sparse.py

    # With IDF stats from Phase 1:
    BG_STATS_PATH=/path/to/phase1/cache_last.pt \\
    PHASE1_CACHE_PATH=/path/to/phase1/cache_last.pt \\
    SYNTH_DATA_PATH=/path/to/phase2_MT.parquet \\
    python examples/qasper2/train/continual_sparse.py

Env vars:
    PHASE1_CACHE_PATH       (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH         (required) — path to synthesized training parquet (Phase 2 task: MT)
    EVAL_DATA_PATH          (optional) — path to evaluation parquet (perplexity in W&B)
    BG_STATS_PATH           (optional) — path to Phase 1 cache_last.pt for IDF stats;
                              defaults to PHASE1_CACHE_PATH if not set
    NUM_TOKENS              — cartridge size (default: -1; used in run name only)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 10)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    MAX_STEPS               — cosine-schedule horizon in optimizer steps (default: 250).
                              Rule of thumb on this dataset: 250 for bsize 64, 500 for bsize 32.
    EVAL_EVERY_N_STEPS      — eval interval in optimizer steps (default: 15)
    SAVE_EVERY_N_STEPS      — checkpoint interval (default: 256)
    TOP_T                   — number of cache positions updated per step (default: 64)
    MOMENTUM_MASKING        — masking strategy: soft | hard | freeze | decouple (default: freeze)
    FREEZE_KEYS             — 1=freeze key cache, 0=update keys (default: 0)
    GRANULARITY             — sparse granularity: global | per_layer | per_head (default: per_head)
                              Must match the GRANULARITY used in Phase 1 (initial_sparse.py).
    IDF_TOP_K               — top-k positions per bg batch counted toward df (default: 128)
    IDF_SMOOTHING           — Laplace smoothing for IDF denominator (default: 1.0)
    RUN_NAME                — W&B run name (default: auto-generated)
    WANDB_GROUP             — W&B group name (default: None). Set this in a sweep
                              to cluster all iterations' runs under one group.
    DISTRIBUTED_BACKEND     — distributed backend (default: gloo)
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.cache import TrainableCache, KVCacheFactory
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import SparseCacheFinetuningConfig
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
# Optional: provide a parquet to track perplexity in W&B alongside training.
# Use a Phase 1 parquet (QA) to monitor forgetting, or Phase 2 (MT) for acquisition.
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
# Optional: path to Phase 1 cache for IDF-weighted slot selection (sparsity stats);
# distinct from PHASE1_CACHE_PATH. Defaults to PHASE1_CACHE_PATH if unset.
BG_STATS_PATH = os.environ.get("BG_STATS_PATH", None)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "-1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
# Cosine schedule horizon. Empirically: 250 fits bsize 64, 500 fits bsize 32 on
# the qasper Phase-2 (MT) dataset at EPOCHS=10. Override via env when changing
# either GLOBAL_BATCH_SIZE or EPOCHS so the LR floor doesn't kick in mid-run.
MAX_STEPS = int(os.environ.get("MAX_STEPS", "550"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "15"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")
TOP_T = int(os.environ.get("TOP_T", "64"))
MOMENTUM_MASKING = os.environ.get("MOMENTUM_MASKING", "freeze")  # soft | hard | freeze | decouple
FREEZE_KEYS = os.environ.get("FREEZE_KEYS", "0") not in ("0", "false", "False")
GRANULARITY = os.environ.get("GRANULARITY", "per_head")          # global | per_layer | per_head
IDF_TOP_K = int(os.environ.get("IDF_TOP_K", "128"))              # top-k per bg batch for df computation
IDF_SMOOTHING = float(os.environ.get("IDF_SMOOTHING", "1.0"))    # Laplace smoothing for IDF
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"qasper_phase2_sparse_{MOMENTUM_MASKING}_top-{TOP_T}_{GRANULARITY}_lr{LR}",
)
# Optional W&B group — lets a sweep cluster all its runs under one group in the UI.
WANDB_GROUP = os.environ.get("WANDB_GROUP", None)

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
        max_steps=MAX_STEPS,
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
    # Perplexity eval (optional) — set EVAL_DATA_PATH to a parquet to enable.
    # Tip: pass the Phase 1 (QA) parquet here to monitor forgetting in real time.
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
    wandb=WandBConfig(
        group=WANDB_GROUP,
        tags=[
            "train", "qasper", "phase2", "sparse",
            f"top-t-{TOP_T}", f"momentum-{MOMENTUM_MASKING}",
            f"idf-top-k-{IDF_TOP_K}", GRANULARITY, f"lr-{LR}",
        ]
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
