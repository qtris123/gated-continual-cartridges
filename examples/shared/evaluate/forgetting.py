"""Standalone loss eval — cartridge OR full-context ICL, from ONE script.

Two modes, selected by EVAL_MODE:
  * cartridge (default): eval a saved cartridge checkpoint on an eval parquet via the
    canonical `cartridges/` path (`LossEvalRunConfig` -> evaluate_perplexity). This is
    the path that produced the cartridge / AM / sparse-grad numbers.
  * icl: full-context ICL baseline (the topic's papers placed in the prompt, NO cartridge)
    on the SAME eval parquet, via `examples/shared/evaluate/icl.py`. Uses the SAME
    soft-CE-vs-teacher-top-k scorer as the cartridge path, so the numbers are directly
    comparable (same ruler).

Usage:
    # cartridge (unchanged)
    CHECKPOINT_PATH=/path/cache_last.pt EVAL_DATA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 python examples/shared/evaluate/forgetting.py

    # ICL full-context, SAME eval file
    EVAL_MODE=icl ICL_TOPIC=QA EVAL_DATA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 python examples/shared/evaluate/forgetting.py
    # writes outputs/<YYYY-MM-DD-HH-MM-SS>-ICL-<topic>/{config.yaml,results.json}

Env vars:
    EVAL_MODE           — cartridge (default) | icl
    EVAL_DATA_PATH      (required) — eval parquet (QA = Phase 1 forgetting, MT = Phase 2 acq)
    MODEL_NAME          — HF model id (default: Qwen/Qwen3-4B-Instruct-2507)
    CARTRIDGES_OUTPUT_DIR — root for run folders (default: $CARTRIDGES_DIR/outputs or ./outputs)
  cartridge mode:
    CHECKPOINT_PATH     (required) — local path to a .pt cartridge checkpoint
    RUN_NAME, BATCH_SIZE
  icl mode:
    ICL_TOPIC           (required) — QA | MT | 'QA+MT' | 'MT+QA' (papers placed in context)
    RUN_NAME            — optional label stored in config/results (default: icl_<topic>)
    PREFILL_CHUNK_SIZE  — chunked-prefill size (default 2048)
    MAX_CONTEXT_TOKENS  — optional context truncation
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pydrantic
import yaml

from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.evaluate import LossEvalRunConfig
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig

EVAL_MODE = os.environ.get("EVAL_MODE", "cartridge").lower()
EVAL_DATA_PATH = os.environ["EVAL_DATA_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")


class KVFromLocal(KVCacheFactory):
    """Load a KV cache from a local checkpoint file (cartridge mode)."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


def _output_root() -> Path:
    root = os.environ.get("CARTRIDGES_OUTPUT_DIR")
    if root:
        return Path(root)
    cartridges_dir = os.environ.get("CARTRIDGES_DIR")
    if cartridges_dir:
        return Path(cartridges_dir) / "outputs"
    return Path("outputs")


def _icl_dataset_tag(topic: str, eval_path: str) -> str:
    """Folder tag after ICL-: prefer topic (QA/MT), else eval parquet stem."""
    tag = topic.strip().replace("+", "-").replace("/", "-") or Path(eval_path).stem
    return "".join(c if (c.isalnum() or c in "-_") else "-" for c in tag)


def _make_icl_run_dir(topic: str, eval_path: str) -> Path:
    time_tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = _output_root() / f"{time_tag}-ICL-{_icl_dataset_tag(topic, eval_path)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _run_icl():
    """Full-context ICL eval (no cartridge) via the shared eval_icl library."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow sibling import
    from eval_icl import run_icl_loss_eval

    topic = os.environ["ICL_TOPIC"]
    prefill = int(os.environ.get("PREFILL_CHUNK_SIZE", "2048"))
    _mct = os.environ.get("MAX_CONTEXT_TOKENS")
    max_context_tokens = int(_mct) if _mct else None
    run_name = os.environ.get("RUN_NAME", f"icl_{topic.replace('+', '-')}")
    run_dir = _make_icl_run_dir(topic, EVAL_DATA_PATH)

    config = {
        "eval_mode": "icl",
        "name": run_name,
        "model_name": MODEL_NAME,
        "eval_data_path": EVAL_DATA_PATH,
        "icl_topic": topic,
        "prefill_chunk_size": prefill,
        "max_context_tokens": max_context_tokens,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "run_dir": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"[icl] run_dir={run_dir}", flush=True)

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

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "eval_mode": "icl",
        "icl_topic": topic,
        "eval_data_path": EVAL_DATA_PATH,
        "model_name": MODEL_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[icl] wrote {run_dir / 'config.yaml'} and {run_dir / 'results.json'}", flush=True)
    return metrics


def _run_cartridge():
    """Canonical cartridge eval (unchanged behavior)."""
    CHECKPOINT_PATH = os.environ["CHECKPOINT_PATH"]
    RUN_NAME = os.environ.get("RUN_NAME", "forgetting_eval")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
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
        wandb=None,  # results are parsed off the "Eval loss" log line
    )
    pydrantic.main(config)


if __name__ == "__main__":
    if EVAL_MODE == "icl":
        _run_icl()
    else:
        _run_cartridge()
