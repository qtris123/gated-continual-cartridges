"""Standalone loss eval — cartridge OR full-context ICL, from ONE script.

Two modes, selected by EVAL_MODE:
  * cartridge (default): eval a saved cartridge checkpoint on an eval parquet via the
    canonical `cartridges/` path (`LossEvalRunConfig` -> evaluate_perplexity). This is
    the path that produced the cartridge / AM / sparse-grad numbers.
  * icl: full-context ICL baseline (the topic's papers placed in the prompt, NO cartridge)
    on the SAME eval parquet, via `examples/qasper2/train/eval_icl.py`. Uses the SAME
    soft-CE-vs-teacher-top-k scorer as the cartridge path, so the numbers are directly
    comparable (same ruler).

Usage:
    # cartridge (unchanged)
    CHECKPOINT_PATH=/path/cache_last.pt EVAL_DATA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 python examples/qasper2/train/eval_forgetting.py

    # ICL full-context, SAME eval file
    EVAL_MODE=icl ICL_TOPIC=QA EVAL_DATA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 python examples/qasper2/train/eval_forgetting.py

Env vars:
    EVAL_MODE           — cartridge (default) | icl
    EVAL_DATA_PATH      (required) — eval parquet (QA = Phase 1 forgetting, MT = Phase 2 acq)
    MODEL_NAME          — HF model id (default: Qwen/Qwen3-4B-Instruct-2507)
  cartridge mode:
    CHECKPOINT_PATH     (required) — local path to a .pt cartridge checkpoint
    RUN_NAME, BATCH_SIZE, WANDB_GROUP, WANDB_DISABLED
  icl mode:
    ICL_TOPIC           (required) — QA | MT | 'QA+MT' | 'MT+QA' (papers placed in context)
    PREFILL_CHUNK_SIZE  — chunked-prefill size (default 2048)
    MAX_CONTEXT_TOKENS  — optional context truncation
"""

import os

import pydrantic

from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.evaluate import LossEvalRunConfig
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig
from cartridges.utils.wandb import WandBConfig

EVAL_MODE = os.environ.get("EVAL_MODE", "cartridge").lower()
EVAL_DATA_PATH = os.environ["EVAL_DATA_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")


class KVFromLocal(KVCacheFactory):
    """Load a KV cache from a local checkpoint file (cartridge mode)."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


def _run_icl():
    """Full-context ICL eval (no cartridge) via the shared eval_icl library."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow sibling import
    from eval_icl import run_icl_loss_eval

    topic = os.environ["ICL_TOPIC"]
    prefill = int(os.environ.get("PREFILL_CHUNK_SIZE", "2048"))
    _mct = os.environ.get("MAX_CONTEXT_TOKENS")
    max_context_tokens = int(_mct) if _mct else None

    metrics = run_icl_loss_eval(
        model_name=MODEL_NAME,
        eval_path=EVAL_DATA_PATH,
        topic=topic,
        prefill_chunk_size=prefill,
        max_context_tokens=max_context_tokens,
    )
    # Match the canonical log line so downstream parsing ("Eval loss - <float>") is identical.
    print(f"Eval loss - {metrics['loss']}", flush=True)
    print(f"[icl] {metrics}", flush=True)
    return metrics


def _run_cartridge():
    """Canonical cartridge eval (unchanged behavior)."""
    CHECKPOINT_PATH = os.environ["CHECKPOINT_PATH"]
    RUN_NAME = os.environ.get("RUN_NAME", "forgetting_eval")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
    WANDB_GROUP = os.environ.get("WANDB_GROUP") or None
    WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "0") in ("1", "true", "True")
    _model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

    config = LossEvalRunConfig(
        name=RUN_NAME,
        model=HFModelConfig(
            pretrained_model_name_or_path=MODEL_NAME,
            model_cls=_model_cls,
        ),
        kv_cache_initializer=KVFromLocal.Config(
            path=CHECKPOINT_PATH,
        ),
        eval=LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(
                    path=EVAL_DATA_PATH,
                    type="local",
                ),
                packed_seq_length=2048,
            ),
            name_for_wandb="qasper_perplexity",
        ),
        batch_size=BATCH_SIZE,
        wandb=None if WANDB_DISABLED else WandBConfig(
            tags=["eval", "forgetting", RUN_NAME],
            group=WANDB_GROUP,
        ),
    )
    pydrantic.main(config)


if __name__ == "__main__":
    if EVAL_MODE == "icl":
        _run_icl()
    else:
        _run_cartridge()
