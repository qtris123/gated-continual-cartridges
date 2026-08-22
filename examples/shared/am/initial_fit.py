"""Phase 1: AM-based cartridge initialization (backprop-free).

Initializes cache from init text via KVFromText, refines values using closed-form
Attention Matching on Phase 1 self-study queries, collects bg_stats.pt, and evaluates.

Usage:
    TEXT_PATH=data/qasper/init_text/qasper_init_512.txt \\
    SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_QA_task_8192.parquet \\
    python examples/shared/am/initial_fit.py
"""

import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pydrantic
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig
from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import (
    SparseCacheFinetuningConfig,
    collect_background_stats,
)
from cartridges.am.initial import refine_cache_am_phase1
from cartridges.train import TrainConfig, LossEvalConfig, CacheAndModel, evaluate_perplexity, save_cache
from cartridges.utils import seed_everything, get_logger
from cartridges.utils.wandb import WandBConfig, prepare_wandb

logger = get_logger(__name__)

TEXT_PATH = os.environ["TEXT_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "512"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
AM_PASSES = int(os.environ.get("AM_PASSES", "3"))
AM_MAX_BATCHES = int(os.environ.get("AM_MAX_BATCHES", "50"))
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-4"))
QUERIES_PER_BATCH = os.environ.get("QUERIES_PER_BATCH", "all_tokens")
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
GRANULARITY = os.environ.get("GRANULARITY", "per_layer")
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "nccl")
SEED = int(os.environ.get("SEED", "42"))
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"qasper_phase1_am_{MODEL_NAME.split('/')[-1]}_{NUM_TOKENS}_{GRANULARITY}",
)

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM


def _collate_first(batch):
    return batch[0]


def _setup_run_dir(config: TrainConfig) -> Path:
    if config.run_dir is None:
        time_tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        script_id = "initial_am"
        run_id = str(uuid.uuid4())
        config.launch_id = f"{time_tag}-{script_id}"
        config.run_id = run_id
        config.run_dir = os.path.join(config.output_dir, config.launch_id, run_id)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(run_dir / "config.yaml"))
    return run_dir


def run_am_phase1(config: TrainConfig):
    seed_everything(config.seed)
    local_rank = "cuda" if torch.cuda.is_available() else "cpu"
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    run_dir = _setup_run_dir(config)

    if config.wandb is not None:
        config.wandb.name = config.name
        prepare_wandb(config.wandb, config.to_dict())
        import wandb

        wandb.log({"phase": 1, "run_dir": str(run_dir)}, step=0)

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    dataset = config.dataset.instantiate(tokenizer=tokenizer, seed=config.seed)
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=_collate_first, num_workers=0,
    )

    model = config.model.instantiate().to(local_rank).to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    attn_config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=(
            model.config.head_dim
            if hasattr(model.config, "head_dim")
            else model.config.hidden_size // model.config.num_attention_heads
        ),
    )

    t_init = time.time()
    initializer = config.kv_cache_initializer.instantiate()
    cache = initializer.initialize_kv_cache(
        tokenizer=tokenizer, model=model, attn_config=attn_config,
    ).to(local_rank)
    init_time = time.time() - t_init
    logger.info(f"KVFromText init completed in {init_time:.1f}s")

    t_am = time.time()
    am_stats = refine_cache_am_phase1(
        cache=cache,
        model=model,
        dataloader=dataloader,
        n_am_passes=AM_PASSES,
        max_batches=AM_MAX_BATCHES,
        ridge_lambda=RIDGE_LAMBDA,
        queries_per_batch=QUERIES_PER_BATCH,
        local_rank=local_rank,
    )
    am_time = time.time() - t_am
    logger.info(f"AM refinement completed in {am_time:.1f}s: {am_stats}")

    # Save timing stats
    timing = {"init_time_s": init_time, "am_time_s": am_time, "am_stats": am_stats}
    torch.save(timing, run_dir / "am_timing.pt")

    if config.wandb is not None:
        import wandb

        log_dict = {
            "phase1/init_time_s": init_time,
            "phase1/am_time_s": am_time,
            "phase1/final_mean_mse": am_stats.get("final_mean_mse", 0.0),
        }
        for pass_rec in am_stats.get("passes", []):
            p = pass_rec.get("pass", 0)
            log_dict[f"phase1/pass_{p}/mean_mse"] = pass_rec.get("mean_mse", 0.0)
            log_dict[f"phase1/pass_{p}/n_batches"] = pass_rec.get("n_batches", 0)
        wandb.log(log_dict, step=0)

    # Collect bg_stats
    sparse_cfg = config.sparse_cache_finetuning
    if sparse_cfg and sparse_cfg.collect_background_stats:
        wrapped = CacheAndModel(cache, model, sparse_config=sparse_cfg)
        bg_path = str(run_dir / "bg_stats.pt")
        collect_background_stats(
            wrapped_model=wrapped,
            cache=cache,
            dataloader=dataloader,
            config=sparse_cfg,
            local_rank=local_rank,
            save_path=bg_path,
            is_ddp=False,
            is_rank_zero=True,
        )

    # Eval perplexity
    if config.loss_evals:
        wrapped = CacheAndModel(cache, model)
        for ds_config, eval_dataset in [
            (le, le.dataset.instantiate(tokenizer=tokenizer, seed=config.seed))
            for le in config.loss_evals
        ]:
            evaluate_perplexity(
                config=config,
                model=wrapped,
                cache=cache,
                eval_dataset=eval_dataset,
                ds_config=ds_config,
                optimizer_step=0,
                epoch=0,
                local_rank=local_rank,
                cache_tuning=True,
            )

    save_cache(config, cache, optimizer_step=0)
    logger.info(f"AM Phase 1 complete. Total AM time: {am_time:.1f}s. Saved to {run_dir}")

    if config.wandb is not None:
        import wandb

        wandb.log(
            {
                "phase1/complete": 1,
                "phase1/cache_path": str(run_dir / "cache_last.pt"),
                "phase1/bg_stats_path": str(run_dir / "bg_stats.pt"),
            },
            step=0,
        )
        wandb.finish()

    return timing


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
    lr=0.0,
    epochs=0,
    global_batch_size=GLOBAL_BATCH_SIZE,
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=SYNTH_DATA_PATH, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
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
    sparse_cache_finetuning=SparseCacheFinetuningConfig(
        enabled=False,
        collect_background_stats=True,
        num_background_batches=1000,
        granularity=GRANULARITY,
    ),
    save_after_training=True,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(
        tags=["train", "qasper", "phase1", "am", GRANULARITY.replace("_", "-")],
        group=os.environ.get("WANDB_GROUP") or None,
        notes=os.environ.get("WANDB_NOTES") or None,
    ) if os.environ.get("WANDB_DISABLED", "0") not in ("1", "true", "True") else None,
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
    seed=SEED,
)


if __name__ == "__main__":
    run_am_phase1(config)
