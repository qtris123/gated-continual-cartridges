"""Phase 2: Continual cartridge update with TF-IDF + Attention Matching (backprop-free).

Loads a Phase 1 cartridge and applies sparse AM updates guided by TF-IDF slot selection.

Execution modes (AM_EXECUTION_MODE):
- per_document (default): one closed-form write per unique MT system_prompt
- legacy_decoupled / decoupled: single global TF-IDF solve (legacy baseline)
- train_loop: optimizer-step-shaped loop via pydrantic.main

Usage:
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \\
    BG_STATS_PATH=/path/to/bg_stats.pt \\
    SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \\
    python examples/qasper2/train/continual_am_sparse.py
"""

import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.am_continual import run_per_document_am_phase2
from cartridges.cache import AttnConfig, TrainableCache, KVCacheFactory
from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import (
    BackgroundAccessTracker,
    CacheTFIDFRanker,
    SparseCacheFinetuningConfig,
)
from cartridges.attention_matching_finetuning import (
    AttentionMatchingFinetuningConfig,
    run_decoupled_tfidf_am_update,
)
from cartridges.train import (
    CacheAndModel,
    TrainConfig,
    LossEvalConfig,
    evaluate_perplexity,
    save_cache,
)
from cartridges.utils import get_logger, seed_everything
from cartridges.utils.wandb import WandBConfig, prepare_wandb


class KVFromLocal(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
OLD_REF_DATA_PATH = os.environ.get("OLD_REF_DATA_PATH") or None
BG_STATS_PATH = os.environ.get("BG_STATS_PATH") or None
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "-1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
EPOCHS = int(os.environ.get("EPOCHS", "10"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "550"))
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "15"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "nccl")
TOP_T = int(os.environ.get("TOP_T", "64"))
GRANULARITY = os.environ.get("GRANULARITY", "per_layer")
IDF_TOP_K = int(os.environ.get("IDF_TOP_K", "128"))
IDF_SMOOTHING = float(os.environ.get("IDF_SMOOTHING", "1.0"))
TARGET_MODE = os.environ.get("TARGET_MODE", "cartridge_plus_doc")
QUERIES_PER_BATCH = os.environ.get("QUERIES_PER_BATCH", "all_tokens")
UPDATE_INTERVAL = int(os.environ.get("UPDATE_INTERVAL", "1"))
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-4"))
RIDGE_SCALE = os.environ.get("RIDGE_SCALE", "spectral")
RIDGE_LAMBDA_MIN = float(os.environ.get("RIDGE_LAMBDA_MIN", "0.0"))
USE_IDF = os.environ.get("USE_IDF", "1") not in ("0", "false", "False")
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"qasper_phase2_am_sparse_top-{TOP_T}_{GRANULARITY}_{TARGET_MODE}",
)
WANDB_GROUP = os.environ.get("WANDB_GROUP", None)
AM_EXECUTION_MODE = os.environ.get("AM_EXECUTION_MODE", "per_document")
if AM_EXECUTION_MODE == "decoupled":
    AM_EXECUTION_MODE = "legacy_decoupled"
AM_DECOUPLED_REF_BATCHES = int(os.environ.get("AM_DECOUPLED_REF_BATCHES", "5"))
AM_COMPUTE_STATS = os.environ.get("AM_COMPUTE_STATS", "1") not in ("0", "false", "False")
OLD_REFERENCE_WEIGHT = float(os.environ.get("OLD_REFERENCE_WEIGHT", "1.0"))
DELTA_WEIGHT = float(os.environ.get("DELTA_WEIGHT", "1e-2"))
SLOT_SELECTION = os.environ.get("SLOT_SELECTION", "tfidf")
IDF_PRIOR_WEIGHT = float(os.environ.get("IDF_PRIOR_WEIGHT", "0.0"))
MIN_TOP_T_PER_LAYER = int(os.environ.get("MIN_TOP_T_PER_LAYER", "1"))
ENABLE_OLD_REFERENCE_GUARD = os.environ.get("ENABLE_OLD_REFERENCE_GUARD", "0") in (
    "1", "true", "True",
)
KEY_MODE = os.environ.get("KEY_MODE", "freeze")
ENABLE_BETA_ENV = os.environ.get("ENABLE_BETA")
ENABLE_BETA = (
    None
    if ENABLE_BETA_ENV is None
    else ENABLE_BETA_ENV in ("1", "true", "True")
)
BETA_FIT_SCOPE = os.environ.get("BETA_FIT_SCOPE", "selected")
MAX_REF_EXAMPLES_PER_DOC = int(os.environ.get("MAX_REF_EXAMPLES_PER_DOC", "32"))
SAVE_AFTER_EACH_DOCUMENT = os.environ.get("SAVE_AFTER_EACH_DOCUMENT", "1") not in (
    "0", "false", "False",
)
OLD_REF_MAX_EXAMPLES = int(os.environ.get("OLD_REF_MAX_EXAMPLES", "64"))
EVAL_QA_PATH = os.environ.get("EVAL_QA_PATH") or None
EVAL_MT_PATH = os.environ.get("EVAL_MT_PATH") or None
# B-ROUTE write-ceiling oracle (opt-in, default OFF).
AM_ORACLE_WRITE = os.environ.get("AM_ORACLE_WRITE", "0") in ("1", "true", "True")
AM_ORACLE_WRITE_ASSIGN = os.environ.get("AM_ORACLE_WRITE_ASSIGN", "mass_ranked")

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM
logger = get_logger(__name__)


def _collate_first(batch):
    return batch[0]


def _setup_run_dir(config: TrainConfig) -> Path:
    if config.run_dir is None:
        time_tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        script_id = "continual_am_sparse"
        run_id = str(uuid.uuid4())
        config.launch_id = f"{time_tag}-{script_id}"
        config.run_id = run_id
        config.run_dir = os.path.join(config.output_dir, config.launch_id, run_id)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(run_dir / "config.yaml"))
    return run_dir


def _oracle_write_kwargs() -> dict:
    """Pass the oracle flags ONLY if the imported cartridges package has them.

    RUNBOOK §6.10: `import cartridges` can resolve to the sibling repo, whose
    config would reject an unknown field and crash every stock run. So the kwarg
    is conditional — and if the oracle was explicitly requested but the field is
    missing, fail loudly instead of silently running the ordinary solve.
    """
    import cartridges

    has_field = "oracle_write" in AttentionMatchingFinetuningConfig.model_fields
    if not has_field:
        if AM_ORACLE_WRITE:
            raise RuntimeError(
                "AM_ORACLE_WRITE=1 but the imported `cartridges` package "
                f"({os.path.dirname(cartridges.__file__)}) has no `oracle_write` field. "
                "Export PYTHONPATH=$CARTRIDGES_DIR:$PYTHONPATH (RUNBOOK §6.10)."
            )
        return {}
    return {
        "oracle_write": AM_ORACLE_WRITE,
        "oracle_write_assign": AM_ORACLE_WRITE_ASSIGN,
    }


def _build_am_config() -> AttentionMatchingFinetuningConfig:
    return AttentionMatchingFinetuningConfig(
        **_oracle_write_kwargs(),
        enabled=True,
        top_t=TOP_T,
        use_idf=USE_IDF and BG_STATS_PATH is not None,
        background_indices_path=BG_STATS_PATH,
        collect_background_stats=True,
        num_background_batches=999999999,
        granularity=GRANULARITY,
        background_top_k_per_batch=IDF_TOP_K,
        idf_smoothing=IDF_SMOOTHING,
        execution_mode=AM_EXECUTION_MODE,
        target_mode=TARGET_MODE,
        enable_old_reference_guard=ENABLE_OLD_REFERENCE_GUARD,
        old_ref_data_path=OLD_REF_DATA_PATH,
        old_ref_max_examples=OLD_REF_MAX_EXAMPLES,
        key_mode=KEY_MODE,
        enable_beta=ENABLE_BETA,
        beta_fit_scope=BETA_FIT_SCOPE,
        max_ref_examples_per_doc=MAX_REF_EXAMPLES_PER_DOC,
        save_after_each_document=SAVE_AFTER_EACH_DOCUMENT,
        queries_per_batch=QUERIES_PER_BATCH,
        update_interval=UPDATE_INTERVAL,
        ridge_lambda=RIDGE_LAMBDA,
        ridge_scale=RIDGE_SCALE,
        ridge_lambda_min=RIDGE_LAMBDA_MIN,
        freeze_keys=KEY_MODE == "freeze",
        decoupled_ref_batches=AM_DECOUPLED_REF_BATCHES,
        compute_update_stats=AM_COMPUTE_STATS,
        old_reference_weight=OLD_REFERENCE_WEIGHT if ENABLE_OLD_REFERENCE_GUARD else 0.0,
        delta_weight=DELTA_WEIGHT,
        slot_selection=SLOT_SELECTION,
        idf_prior_weight=IDF_PRIOR_WEIGHT,
        min_top_t_per_layer=MIN_TOP_T_PER_LAYER,
    )


def _init_model_and_cache(config: TrainConfig, local_rank):
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    model = config.model.instantiate().to(local_rank).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    attn_config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=(
            model.config.head_dim
            if hasattr(model.config, "head_dim")
            else model.config.hidden_size // model.config.num_attention_heads
        ),
    )

    initializer = config.kv_cache_initializer.instantiate()
    cache = initializer.initialize_kv_cache(
        tokenizer=tokenizer,
        model=model,
        attn_config=attn_config,
    ).to(local_rank)

    return tokenizer, model, cache, attn_config


def run_decoupled_phase2(config: TrainConfig):
    seed_everything(config.seed)
    local_rank = "cuda" if torch.cuda.is_available() else "cpu"
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    run_dir = _setup_run_dir(config)
    tokenizer, model, cache, attn_config = _init_model_and_cache(config, local_rank)

    dataset = config.dataset.instantiate(tokenizer=tokenizer, seed=config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_collate_first,
        num_workers=0,
    )
    old_dataloader = None
    if OLD_REF_DATA_PATH and config.attention_matching_finetuning.old_reference_weight > 0:
        old_dataset = TrainDataset.Config(
            data_sources=[DataSource(path=OLD_REF_DATA_PATH, type="local")],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ).instantiate(tokenizer=tokenizer, seed=config.seed)
        old_dataloader = DataLoader(
            old_dataset,
            batch_size=1,
            collate_fn=_collate_first,
            num_workers=0,
        )

    am_ft = config.attention_matching_finetuning
    bg_tracker = None
    if am_ft.background_indices_path is not None:
        bg_tracker = BackgroundAccessTracker(
            num_batches=am_ft.num_background_batches,
            granularity=am_ft.granularity,
        )
        bg_tracker.load(am_ft.background_indices_path)

    tfidf_ranker = CacheTFIDFRanker(
        background_tracker=bg_tracker,
        use_idf=am_ft.use_idf,
        smoothing=am_ft.idf_smoothing,
        top_k_per_batch=am_ft.background_top_k_per_batch,
        granularity=am_ft.granularity,
    )

    wrapped = CacheAndModel(cache, model, am_config=am_ft).to(local_rank)
    t0 = time.time()
    am_stats, ranking_info = run_decoupled_tfidf_am_update(
        cache=cache,
        wrapped_model=wrapped,
        dataloader=dataloader,
        tfidf_ranker=tfidf_ranker,
        config=am_ft,
        n_layers=attn_config.n_layers,
        n_kv_heads=attn_config.n_heads,
        head_dim=attn_config.head_dim,
        local_rank=local_rank,
        step=1,
        old_dataloader=old_dataloader,
    )
    wall_clock = time.time() - t0

    torch.save(
        {
            "am_stats": am_stats,
            "ranking_info": ranking_info,
            "wall_clock_s": wall_clock,
        },
        run_dir / "decoupled_am_stats.pt",
    )

    if config.loss_evals:
        eval_wrapped = CacheAndModel(cache, model)
        for ds_config, eval_dataset in [
            (le, le.dataset.instantiate(tokenizer=tokenizer, seed=config.seed))
            for le in config.loss_evals
        ]:
            evaluate_perplexity(
                config=config,
                model=eval_wrapped,
                cache=cache,
                eval_dataset=eval_dataset,
                ds_config=ds_config,
                optimizer_step=1,
                epoch=0,
                local_rank=local_rank,
                cache_tuning=True,
            )

    save_cache(config, cache, optimizer_step=1)
    wrapped.remove_hooks()
    logger.info(
        f"Decoupled AM Phase 2 complete in {wall_clock:.1f}s. Saved to {run_dir}"
    )
    return {"am_stats": am_stats, "wall_clock_s": wall_clock, "run_dir": str(run_dir)}


def run_per_document_phase2(config: TrainConfig):
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

        wandb.log(
            {
                "phase": 2,
                "run_dir": str(run_dir),
                "phase2/phase1_cache": PHASE1_CACHE_PATH,
                "phase2/bg_stats": BG_STATS_PATH or "",
                "phase2/mt_data": SYNTH_DATA_PATH,
            },
            step=0,
        )

    tokenizer, model, cache, attn_config = _init_model_and_cache(config, local_rank)
    am_ft = config.attention_matching_finetuning

    if am_ft.enable_old_reference_guard and not am_ft.old_ref_data_path:
        raise ValueError(
            "ENABLE_OLD_REFERENCE_GUARD=1 requires OLD_REF_DATA_PATH (QA parquet)"
        )

    bg_tracker = None
    if am_ft.background_indices_path is not None:
        bg_tracker = BackgroundAccessTracker(
            num_batches=am_ft.num_background_batches,
            granularity=am_ft.granularity,
        )
        bg_tracker.load(am_ft.background_indices_path)

    tfidf_ranker = CacheTFIDFRanker(
        background_tracker=bg_tracker,
        use_idf=am_ft.use_idf,
        smoothing=am_ft.idf_smoothing,
        top_k_per_batch=am_ft.background_top_k_per_batch,
        granularity=am_ft.granularity,
    )

    wrapped = CacheAndModel(cache, model, am_config=am_ft).to(local_rank)

    def _save(doc_step: int, slug: str):
        save_cache(config, cache, optimizer_step=doc_step)
        doc_ckpt = run_dir / f"cache-after-{slug}.pt"
        cache.save(str(doc_ckpt))

    t0 = time.time()
    aggregate = run_per_document_am_phase2(
        cache=cache,
        wrapped_model=wrapped,
        model=model,
        tokenizer=tokenizer,
        mt_data_path=SYNTH_DATA_PATH,
        tfidf_ranker=tfidf_ranker,
        config=am_ft,
        n_layers=attn_config.n_layers,
        n_kv_heads=attn_config.n_heads,
        head_dim=attn_config.head_dim,
        local_rank=local_rank,
        run_dir=run_dir,
        save_cache_fn=_save if am_ft.save_after_each_document else None,
    )
    wall_clock = time.time() - t0
    aggregate["wall_clock_s"] = wall_clock
    torch.save(aggregate, run_dir / "per_document_am_stats.pt")

    if config.wandb is not None:
        import wandb

        for doc_rec in aggregate.get("per_document", []):
            step = int(doc_rec.get("doc_index", 0)) + 1
            wandb.log(
                {
                    "phase2/doc_index": doc_rec.get("doc_index", 0),
                    "phase2/mean_mse": doc_rec.get("mean_mse", 0.0),
                    "phase2/n_queries": doc_rec.get("n_queries", 0),
                    "phase2/n_conversations": doc_rec.get("n_conversations", 0),
                    "phase2/prefill_s": doc_rec.get("timing_s", {}).get("prefill_s", 0.0),
                    "phase2/doc_total_s": doc_rec.get("timing_s", {}).get("total_s", 0.0),
                    "phase2/slug": doc_rec.get("slug", ""),
                },
                step=step,
            )
        wandb.log(
            {
                "phase2/n_documents": aggregate.get("n_documents", 0),
                "phase2/wall_clock_s": wall_clock,
                "phase2/old_ref_batches": aggregate.get("old_ref_batches", 0),
            },
            step=max(aggregate.get("n_documents", 1), 1),
        )

    if config.loss_evals:
        eval_wrapped = CacheAndModel(cache, model)
        eval_metrics = {}
        for ds_config, eval_dataset in [
            (le, le.dataset.instantiate(tokenizer=tokenizer, seed=config.seed))
            for le in config.loss_evals
        ]:
            metrics = evaluate_perplexity(
                config=config,
                model=eval_wrapped,
                cache=cache,
                eval_dataset=eval_dataset,
                ds_config=ds_config,
                optimizer_step=aggregate.get("n_documents", 1),
                epoch=0,
                local_rank=local_rank,
                cache_tuning=True,
            )
            if metrics:
                eval_metrics[metrics["name"]] = metrics
        aggregate["eval_metrics"] = eval_metrics

    value_norms = _value_norm_summary(cache)
    aggregate["value_norms"] = value_norms
    # Persist lightweight summary for free-GPU A/B launcher.
    import json

    summary_path = run_dir / "phase2_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": config.name,
                "run_dir": str(run_dir),
                "n_documents": aggregate.get("n_documents", 0),
                "wall_clock_s": wall_clock,
                "mean_mse_last_doc": (
                    (aggregate.get("per_document") or [{}])[-1].get("mean_mse")
                    if aggregate.get("per_document")
                    else None
                ),
                "eval_metrics": aggregate.get("eval_metrics", {}),
                "value_norms": {
                    k: value_norms[k]
                    for k in ("global_max_abs", "L0_max_abs", "L18_max_abs", "L35_max_abs")
                    if k in value_norms
                },
                "reg": {
                    "ridge_scale": RIDGE_SCALE,
                    "ridge_lambda": RIDGE_LAMBDA,
                    "ridge_lambda_min": RIDGE_LAMBDA_MIN,
                    "delta_weight": DELTA_WEIGHT,
                },
            },
            indent=2,
        )
    )

    save_cache(config, cache, optimizer_step=aggregate.get("n_documents", 1) or 1)
    wrapped.remove_hooks()
    logger.info(
        "Per-document AM Phase 2 complete in %.1fs (%d documents). Saved to %s",
        wall_clock,
        aggregate.get("n_documents", 0),
        run_dir,
    )

    if config.wandb is not None:
        import wandb

        log_payload = {
            "phase2/complete": 1,
            "phase2/cache_path": str(run_dir / "cache_last.pt"),
            "phase2/value_global_max_abs": value_norms.get("global_max_abs", 0.0),
        }
        for name, m in aggregate.get("eval_metrics", {}).items():
            log_payload[f"phase2/eval_{name}_loss"] = m["loss"]
            log_payload[f"phase2/eval_{name}_ppl"] = m["perplexity"]
        wandb.log(
            log_payload,
            step=max(aggregate.get("n_documents", 1), 1),
        )
        wandb.finish()

    return aggregate


def _build_loss_evals() -> list:
    """Build QA (forgetting) and/or MT (acquisition) eval configs."""
    evals = []
    qa_path = EVAL_QA_PATH
    mt_path = EVAL_MT_PATH
    # Backward compat: single EVAL_DATA_PATH → MT if no dual paths set.
    if EVAL_DATA_PATH and not qa_path and not mt_path:
        mt_path = EVAL_DATA_PATH
    if qa_path:
        evals.append(
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=qa_path, type="local"),
                    packed_seq_length=2048,
                ),
                name_for_wandb="qa_forgetting",
            )
        )
    if mt_path:
        evals.append(
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=mt_path, type="local"),
                    packed_seq_length=2048,
                ),
                name_for_wandb="mt_acquisition",
            )
        )
    return evals


def _value_norm_summary(cache: TrainableCache) -> dict:
    summary = {}
    with torch.no_grad():
        for layer_idx, v in enumerate(cache.trainable_values):
            absmax = float(v.detach().float().abs().max().item())
            summary[f"L{layer_idx}_max_abs"] = absmax
        if cache.trainable_values:
            summary["global_max_abs"] = max(
                summary[k] for k in summary if k.endswith("_max_abs")
            )
    return summary


config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromLocal.Config(path=PHASE1_CACHE_PATH),
    optimizer="adam",
    lr=0.0,
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,
    max_optimizer_steps=MAX_STEPS,
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=SYNTH_DATA_PATH, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    attention_matching_finetuning=_build_am_config(),
    sparse_cache_finetuning=SparseCacheFinetuningConfig(enabled=False),
    loss_eval_every_n_steps=EVAL_EVERY_N_STEPS if (EVAL_DATA_PATH or EVAL_QA_PATH or EVAL_MT_PATH) else None,
    loss_evals=_build_loss_evals(),
    save_every_n_steps=SAVE_EVERY_N_STEPS,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(
        group=WANDB_GROUP,
        tags=[
            "train", "qasper", "phase2", "am-sparse",
            f"top-t-{TOP_T}", f"target-{TARGET_MODE}",
            f"idf-top-k-{IDF_TOP_K}", GRANULARITY,
            f"key-{KEY_MODE}", f"exec-{AM_EXECUTION_MODE}",
            f"ridge-{RIDGE_SCALE}", f"lammin-{RIDGE_LAMBDA_MIN}",
            f"delta-{DELTA_WEIGHT}",
        ] + (
            ["oracle", "diagnostic", f"oracle-write-{AM_ORACLE_WRITE_ASSIGN}"]
            if AM_ORACLE_WRITE
            else []
        ),
        notes=os.environ.get("WANDB_NOTES") or None,
    ) if os.environ.get("WANDB_DISABLED", "0") not in ("1", "true", "True") else None,
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    mode = config.attention_matching_finetuning.execution_mode
    if mode == "per_document":
        run_per_document_phase2(config)
    elif mode in ("legacy_decoupled", "decoupled", "solve_only", "single_pass"):
        run_decoupled_phase2(config)
    elif mode == "train_loop":
        pydrantic.main(config)
    else:
        raise ValueError(f"Unknown AM_EXECUTION_MODE: {mode}")
