"""Evaluate a saved cartridge on up to five held-out phase sets, without writing.

Multi-stage runs need a "before" row: stage-3 acquisition on SA is meaningless
without the SA loss of the cartridge the stage started from. The write path only
reports evals after it has already modified the cartridge, so this reuses the
same `evaluate_perplexity` call to keep the numbers comparable with
`phase2_summary.json` from the write runs.

Usage:
    CACHE_PATH=outputs/.../cache_last.pt \\
    EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \\
    EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \\
    EVAL_SA_PATH=data/qasper/eval/qasper_eval_SA.parquet \\
    python examples/shared/evaluate/cartridge_perplexity.py
"""

import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import CacheAndModel, LossEvalConfig, evaluate_perplexity
from cartridges.utils import get_logger

logger = get_logger(__name__)

CACHE_PATH = os.environ["CACHE_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
OUT_JSON = os.environ.get("OUT_JSON") or None


@dataclass
class _StubConfig:
    """`evaluate_perplexity` only reads `.wandb`; None disables logging."""

    wandb: None = None


def main() -> None:
    device = torch.device("cuda")

    model_config = HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=(
            FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = model_config.instantiate().to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    cache = TrainableCache.from_pretrained(CACHE_PATH, device="cuda").to(device)
    n_slots = cache.trainable_keys[0].size(2)
    logger.info(
        "Loaded cartridge %s: %d layers, %d kv heads, %d trainable slots",
        CACHE_PATH,
        len(cache.trainable_keys),
        cache.trainable_keys[0].size(1),
        n_slots,
    )

    wrapped = CacheAndModel(cache, model)
    results = {}
    phase_paths = [os.environ.get(f"EVAL_P{i}_PATH") for i in range(1, 6)]
    if any(phase_paths):
        evals = [
            (path, os.environ.get(f"EVAL_P{i}_NAME", f"p{i}"))
            for i, path in enumerate(phase_paths, start=1)
        ]
    else:
        evals = [
            (os.environ.get("EVAL_QA_PATH"), "qa_forgetting"),
            (os.environ.get("EVAL_MT_PATH"), "mt_acquisition"),
            (os.environ.get("EVAL_SA_PATH"), "sa_acquisition"),
        ]

    for path, metric_name in evals:
        if not path:
            continue
        ds_config = LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path=path, type="local"),
                packed_seq_length=2048,
            ),
            name_for_wandb=metric_name,
        )
        metrics = evaluate_perplexity(
            config=_StubConfig(),
            model=wrapped,
            cache=cache,
            eval_dataset=ds_config.dataset.instantiate(tokenizer=tokenizer, seed=0),
            ds_config=ds_config,
            optimizer_step=0,
            epoch=0,
            local_rank=device,
            cache_tuning=True,
        )
        if metrics:
            results[metrics["name"]] = metrics

    summary = {"cache_path": CACHE_PATH, "n_slots": n_slots, "eval_metrics": results}
    print(json.dumps(summary, indent=2))
    if OUT_JSON:
        os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
        with open(OUT_JSON, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
