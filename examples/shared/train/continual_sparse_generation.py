"""Phase 2: Continual update with TF-IDF sparse cache finetuning.

Loads a Phase 1 cartridge (trained via initial_sparse.py) and continues training
on patients 11-20 data. Uses sparse gradient masking (TF-IDF) to update only the
most task-relevant cache positions, protecting Phase 1 knowledge.

The BG_STATS_PATH should point to the cache_last.pt from the Phase 1 run produced
by train/initial_sparse.py — it contains the background position statistics needed
to compute IDF-weighted slot selection.

Usage:
    PHASE1_CACHE_PATH=/path/to/phase1/cache_last.pt \\
    SYNTH_DATA_PATH=/path/to/phase2.parquet \\
    python examples/shared/train/continual_sparse_generation.py

    # With IDF stats from Phase 1:
    BG_STATS_PATH=/path/to/phase1/cache_last.pt \\
    PHASE1_CACHE_PATH=/path/to/phase1/cache_last.pt \\
    SYNTH_DATA_PATH=/path/to/phase2.parquet \\
    python examples/shared/train/continual_sparse_generation.py

Env vars:
    PHASE1_CACHE_PATH       (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH         (required) — path to synthesized training parquet (patients 11-20)
    BG_STATS_PATH           (optional) — path to Phase 1 cache_last.pt for IDF stats;
                              defaults to PHASE1_CACHE_PATH if not set
    NUM_TOKENS              — cartridge size (default: -1; used in run name only)
    MODEL_NAME              — HF model name (default: Qwen/Qwen3-4B-Instruct-2507)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 10)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    EVAL_EVERY_N_STEPS      — eval interval in optimizer steps (default: 15)
    SAVE_EVERY_N_STEPS      — checkpoint interval (default: 256)
    PATIENT_IDS             — patient range for eval: "1-10" or "11-20" (default: "1-10")
    TOP_T                   — number of cache positions updated per step (default: 64)
    MOMENTUM_MASKING        — masking strategy: soft | hard | freeze | decouple (default: freeze)
    FREEZE_KEYS             — 1=freeze key cache, 0=update keys (default: 0)
    GRANULARITY             — sparse granularity: global | per_layer | per_head (default: per_head)
                              Must match the GRANULARITY used in Phase 1 (initial_sparse.py).
    IDF_TOP_K               — top-k positions per bg batch counted toward df (default: 128)
    IDF_SMOOTHING           — Laplace smoothing for IDF denominator (default: 1.0)
    RUN_NAME                — W&B run name (default: auto-generated)
    DISTRIBUTED_BACKEND     — distributed backend (default: gloo)
"""

import math
import os

import pandas as pd
import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.cache import TrainableCache, KVCacheFactory
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import SparseCacheFinetuningConfig
from cartridges.train import TrainConfig, GenerationEvalConfig, LossEvalConfig, CosWithWarmup
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.utils.wandb import WandBConfig


class KVFromLocal(KVCacheFactory):
    """Initialize a KV cache from a local checkpoint file."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
# Optional: provide a parquet to also track perplexity alongside MCQ accuracy.
# Use a Phase 1 parquet (patients 1-10) to monitor forgetting, or Phase 2 (11-20) for acquisition.
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
# Optional: path to Phase 1 cache for IDF-weighted slot selection (sparsity stats); distinct from PHASE1_CACHE_PATH.
BG_STATS_PATH = os.environ.get("BG_STATS_PATH", None)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "-1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
PATIENT_IDS = os.environ.get("PATIENT_IDS", "1-10")
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "15"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")
TOP_T = int(os.environ.get("TOP_T", "64"))
MOMENTUM_MASKING = os.environ.get("MOMENTUM_MASKING", "freeze")   # soft | hard | freeze | decouple
FREEZE_KEYS = os.environ.get("FREEZE_KEYS", "0") not in ("0", "false", "False")
GRANULARITY = os.environ.get("GRANULARITY", "per_head")            # global | per_layer | per_head
IDF_TOP_K = int(os.environ.get("IDF_TOP_K", "128"))               # top-k per bg batch for df computation
IDF_SMOOTHING = float(os.environ.get("IDF_SMOOTHING", "1.0"))     # Laplace smoothing for IDF
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"longhealth_phase2_sparse_{MOMENTUM_MASKING}_top-{TOP_T}_{GRANULARITY}_lr{LR}",
)

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

_PATIENTS = {
    "1-10": [f"patient_{i:02d}" for i in range(1, 11)],
    "11-20": [f"patient_{i:02d}" for i in range(11, 21)],
}

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
        max_steps=250, # for bsize 64 on longehealth, 500 for bsize 32 on longhealth
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
    # Tip: pass patients 1-10 parquet here to monitor Phase 1 forgetting in real time.
    loss_eval_every_n_steps=EVAL_EVERY_N_STEPS if EVAL_DATA_PATH else None,
    loss_evals=(
        [
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=EVAL_DATA_PATH, type="local"),
                    packed_seq_length=2048,
                ),
                name_for_wandb="longhealth_perplexity",
            )
        ]
        if EVAL_DATA_PATH
        else []
    ),
    # Generation eval (MCQ accuracy) — always active
    generate_eval_every_n_steps=EVAL_EVERY_N_STEPS,
    generate_evals=[
        GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=_PATIENTS[PATIENT_IDS],
                max_questions=100,
                include_diagnosis=True,
                cot=True,
            ),
            name_for_wandb="longhealth_accuracy",
            generate_max_new_tokens=512,
            batch_size=16,
            temperature=0.3,
        )
    ],
    save_every_n_steps=SAVE_EVERY_N_STEPS,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(
        tags=[
            "train", "longhealth", "phase2", "sparse",
            f"top-t-{TOP_T}", f"momentum-{MOMENTUM_MASKING}",
            f"idf-top-k-{IDF_TOP_K}", GRANULARITY, f"lr-{LR}",
        ]
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
