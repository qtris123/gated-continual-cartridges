"""Phase 2 entry point: one closed-form AM write per document.

``run_am_continual`` is what ``AMContinualConfig.run`` calls. It owns the run:
model + cartridge init, the document loop, the per-document checkpoints, and the
optional perplexity evals. There is no optimizer and no train loop anywhere in
it -- the write is closed-form.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cartridges.am.components.queries import (
    AMQueryAccumulator,
    build_reference_dataloader,
    cleanup_reference_parquet,
    group_conversations_by_document,
    load_conversations,
    load_old_reference_bank,
)
from cartridges.am.continual.config import AMContinualConfig, AMStages
from cartridges.am.continual.write import apply_document_am_write_to_cache
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.sparse_cache_finetuning import CacheTFIDFRanker
from cartridges.train import CacheAndModel, evaluate_perplexity, save_cache
from cartridges.utils import get_logger, seed_everything

logger = get_logger(__name__)


def _doc_slug(system_prompt: str, index: int) -> str:
    digest = hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:8]
    return f"doc-{index:03d}-{digest}"


def _collect_old_reference_bank(
    cache: nn.Module,
    wrapped_model: nn.Module,
    tokenizer,
    stages: AMStages,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[Optional[AMQueryAccumulator], Optional[dict], int]:
    """Collect the old-task (QA) queries and their pre-write attention outputs.

    The bank is captured ONCE, before any document is written, so the guard
    holds the cartridge to the behaviour it had at the start of Phase 2 rather
    than to whatever the previous document left behind.
    """
    objective_cfg = stages.objective.config
    old_conversations = load_old_reference_bank(
        objective_cfg.old_ref_data_path,
        max_examples=objective_cfg.old_ref_max_examples,
    )
    old_loader, old_tmp = build_reference_dataloader(
        old_conversations,
        tokenizer,
        seed=0,
    )
    try:
        old_query_acc, _, old_batch_count = stages.queries.collect(
            wrapped_model,
            cache,
            old_loader,
            granularity=stages.slots.config.granularity,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            device=device,
            max_batches=stages.queries.config.ref_batch_limit,
        )
    finally:
        cleanup_reference_parquet(old_tmp)

    old_target_bank: dict[tuple[int, int], torch.Tensor] = {}
    for layer_idx in range(n_layers):
        if not old_query_acc._queries[layer_idx]:
            continue
        n_q_heads = old_query_acc._queries[layer_idx][0].shape[1]
        if getattr(cache, "_num_frozen_tokens", 0) > 0:
            layer_keys = torch.cat(
                [cache.frozen_keys[layer_idx], cache.trainable_keys[layer_idx]],
                dim=2,
            )
            layer_values = torch.cat(
                [cache.frozen_values[layer_idx], cache.trainable_values[layer_idx]],
                dim=2,
            )
        else:
            layer_keys = cache.trainable_keys[layer_idx]
            layer_values = cache.trainable_values[layer_idx]
        layer_beta = cache.get_cartridge_beta(layer_idx)
        for head_idx in range(n_kv_heads):
            old_queries = old_query_acc.get_layer_head_queries(
                layer_idx,
                head_idx,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
            ).to(device=device, dtype=layer_keys.dtype)
            head_beta = layer_beta[0, head_idx] if layer_beta is not None else None
            old_target_bank[(layer_idx, head_idx)] = stages.teacher.targets(
                old_queries,
                layer_keys[0, head_idx],
                layer_values[0, head_idx],
                head_dim,
                attention_bias=head_beta,
                rope_theta=stages.rope_theta,
            ).detach().cpu()

    return old_query_acc, old_target_bank, old_batch_count


def run_documents(
    cache: nn.Module,
    wrapped_model: nn.Module,
    model: nn.Module,
    tokenizer,
    document_data_path: str,
    tfidf_ranker: CacheTFIDFRanker,
    stages: AMStages,
    attn_config: AttnConfig,
    device: torch.device,
    run_dir: Optional[Path] = None,
    save_cache_fn=None,
) -> dict:
    """Run one closed-form AM write per unique document (system_prompt group)."""
    n_layers = attn_config.n_layers
    n_kv_heads = attn_config.n_heads
    head_dim = attn_config.head_dim
    granularity = stages.slots.config.granularity

    was_training = wrapped_model.training
    wrapped_model.eval()

    conversations = load_conversations(document_data_path)
    doc_groups = group_conversations_by_document(conversations)
    logger.info(
        "Per-document AM: %d unique documents in %s (teacher dataset=%s, "
        "qasper_topic=%s, quality_phase=%s)",
        len(doc_groups),
        document_data_path,
        stages.teacher.config.dataset,
        stages.teacher.config.qasper_topic,
        stages.teacher.config.quality_phase,
    )

    # Resolve every document prompt up front: a title that does not exist in the
    # configured dataset/phase must fail before the first write lands.
    doc_prompts = {
        document_id: stages.teacher.document_prompt(convs)
        for document_id, convs in doc_groups.items()
    }

    old_query_acc: Optional[AMQueryAccumulator] = None
    old_target_bank: Optional[dict[tuple[int, int], torch.Tensor]] = None
    old_batch_count = 0
    objective_cfg = stages.objective.config
    if objective_cfg.enable_old_reference_guard and objective_cfg.old_ref_data_path:
        old_query_acc, old_target_bank, old_batch_count = _collect_old_reference_bank(
            cache,
            wrapped_model,
            tokenizer,
            stages,
            n_layers,
            n_kv_heads,
            head_dim,
            device,
        )

    # MECH-007 / RUNBOOK §9c: opt-in seed offset for the per-document reference draw.
    seed_offset = stages.queries.config.seed_offset
    if seed_offset < 0:
        raise ValueError(
            f"seed_offset must be >= 0, got {seed_offset}: the per-document draw "
            "is seeded by `doc_idx + seed_offset` and a negative offset would "
            "alias document indices onto each other."
        )
    if seed_offset:
        logger.info(
            "MECH-007: per-document reference draw seeded by doc_idx + %d "
            "(max_ref_examples_per_doc=%s)",
            seed_offset,
            stages.queries.config.max_ref_examples_per_doc,
        )

    per_doc_stats = []
    t_total = time.time()

    for doc_idx, (document_id, doc_conversations) in enumerate(doc_groups.items()):
        system_prompt = doc_prompts[document_id]
        if not system_prompt.strip():
            logger.warning("Skipping empty system_prompt at doc index %d", doc_idx)
            continue

        slug = _doc_slug(document_id, doc_idx)
        logger.info(
            "Document %d/%d (%s): %d conversations",
            doc_idx + 1,
            len(doc_groups),
            slug,
            len(doc_conversations),
        )

        doc_loader, doc_tmp = stages.queries.build_loader(
            doc_conversations,
            tokenizer,
            seed=stages.queries.draw_seed(doc_idx),
        )

        t_doc = time.time()
        try:
            t_prefill = time.time()
            doc_kv = stages.teacher.prefill(
                model,
                tokenizer,
                system_prompt,
                attn_config=cache.config,
                device=device,
                cartridge_cache=cache,
            )
            prefill_s = time.time() - t_prefill

            query_acc, _, batch_count = stages.queries.collect(
                wrapped_model,
                cache,
                doc_loader,
                granularity=granularity,
                n_layers=n_layers,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                device=device,
                max_batches=len(doc_loader),
            )

            grad_mask, ranking_info = stages.slots.select(
                query_acc.get_access_scores(),
                tfidf_ranker,
                cache=cache,
                old_access_scores=(
                    old_query_acc.get_access_scores()
                    if old_query_acc is not None
                    else None
                ),
                step=doc_idx + 1,
            )

            # LIT-006 / MECH-006: on-policy, layer-sequential re-extraction. The
            # hook is built ONLY when the knob is on, and `write` refuses to run
            # with the knob on and no hook.
            onpolicy_refresh_fn = None
            if stages.queries.config.onpolicy_layers > 0:

                def onpolicy_refresh_fn(layer_idx: int, group_size: int):
                    """Re-extract reference queries from the UPDATED cartridge.

                    Layers `< layer_idx` have already been written, so this
                    forward pass sees the perturbed residual stream and produces
                    the queries the remaining layers will actually be asked.
                    """
                    fresh_acc, _, _ = stages.queries.collect(
                        wrapped_model,
                        cache,
                        doc_loader,
                        granularity=granularity,
                        n_layers=n_layers,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        device=device,
                        max_batches=len(doc_loader),
                    )
                    fresh_kv = None
                    if stages.queries.config.onpolicy_refresh_doc_kv:
                        fresh_kv = stages.teacher.prefill(
                            model,
                            tokenizer,
                            system_prompt,
                            attn_config=cache.config,
                            device=device,
                            cartridge_cache=cache,
                        )
                    return fresh_acc, fresh_kv

            am_stats = apply_document_am_write_to_cache(
                cache=cache,
                mask=grad_mask,
                query_accumulator=query_acc,
                doc_kv=doc_kv,
                stages=stages,
                n_layers=n_layers,
                head_dim=head_dim,
                old_query_accumulator=old_query_acc,
                old_target_bank=old_target_bank,
                onpolicy_refresh_fn=onpolicy_refresh_fn,
            )
            am_stats.step = doc_idx + 1
            doc_wall = time.time() - t_doc

            doc_record = {
                "doc_index": doc_idx,
                "slug": slug,
                "n_conversations": len(doc_conversations),
                "n_ref_batches": batch_count,
                "mean_mse": am_stats.mean_mse,
                "n_queries": am_stats.n_queries,
                "timing_s": {
                    "prefill_s": prefill_s,
                    "total_s": doc_wall,
                },
                "extra": dict(am_stats.extra or {}),
            }
            per_doc_stats.append(doc_record)

            if run_dir is not None:
                torch.save(
                    {
                        "doc_record": doc_record,
                        "ranking_info": ranking_info,
                        "am_stats": am_stats,
                    },
                    run_dir / f"am_doc_{slug}.pt",
                )

            if save_cache_fn is not None:
                save_cache_fn(doc_idx + 1, slug)

            logger.info(
                "Document %s complete: mean_mse=%.6f queries=%d time=%.1fs",
                slug,
                am_stats.mean_mse,
                am_stats.n_queries,
                doc_wall,
            )
        finally:
            cleanup_reference_parquet(doc_tmp)

    if was_training:
        wrapped_model.train()

    aggregate = {
        "n_documents": len(per_doc_stats),
        "per_document": per_doc_stats,
        "old_ref_batches": old_batch_count,
        "wall_clock_s": time.time() - t_total,
    }
    if seed_offset:
        # Provenance only, and only when the knob is on, so a stock
        # `per_document_am_stats.pt` keeps exactly its historical keys.
        aggregate["seed_offset"] = seed_offset
    logger.info(
        "Per-document AM Phase 2 complete: %d documents in %.1fs",
        len(per_doc_stats),
        aggregate["wall_clock_s"],
    )
    return aggregate


def _init_model_and_cache(config: AMContinualConfig, device):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    model = config.model.instantiate().to(device).to(torch.bfloat16)
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

    cache = config.kv_cache_initializer.instantiate().initialize_kv_cache(
        tokenizer=tokenizer,
        model=model,
        attn_config=attn_config,
    ).to(device)

    return tokenizer, model, cache, attn_config


def _value_norm_summary(cache: TrainableCache) -> dict:
    summary = {}
    with torch.no_grad():
        for layer_idx, v in enumerate(cache.trainable_values):
            summary[f"L{layer_idx}_max_abs"] = float(v.detach().float().abs().max().item())
        if cache.trainable_values:
            summary["global_max_abs"] = max(
                summary[k] for k in summary if k.endswith("_max_abs")
            )
    return summary


def run_am_continual(config: AMContinualConfig) -> dict:
    """Write every document of ``config.document_data_path`` into the cartridge."""
    seed_everything(config.seed)

    device = config.device if torch.cuda.is_available() else "cpu"
    if "LOCAL_RANK" in os.environ:
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device)

    if config.run_dir is None:
        raise ValueError(
            "AMContinualConfig.run_dir is unset. Launch through `pydrantic.main`, "
            "which derives it from `output_dir`, or set it explicitly."
        )
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = config.build_stages()
    objective_cfg = stages.objective.config
    if objective_cfg.enable_old_reference_guard and not objective_cfg.old_ref_data_path:
        raise ValueError(
            "objective.enable_old_reference_guard=True requires "
            "objective.old_ref_data_path (a QA parquet)"
        )

    tokenizer, model, cache, attn_config = _init_model_and_cache(config, device)
    tfidf_ranker = stages.slots.build_ranker()
    wrapped = CacheAndModel(cache, model, capture_queries=True).to(device)

    def save_after_document(doc_step: int, slug: str):
        save_cache(config, cache, optimizer_step=doc_step)
        cache.save(str(run_dir / f"cache-after-{slug}.pt"))

    t0 = time.time()
    aggregate = run_documents(
        cache=cache,
        wrapped_model=wrapped,
        model=model,
        tokenizer=tokenizer,
        document_data_path=config.document_data_path,
        tfidf_ranker=tfidf_ranker,
        stages=stages,
        attn_config=attn_config,
        device=device,
        run_dir=run_dir,
        save_cache_fn=save_after_document if config.save_after_each_document else None,
    )
    wall_clock = time.time() - t0
    aggregate["wall_clock_s"] = wall_clock
    torch.save(aggregate, run_dir / "per_document_am_stats.pt")

    n_documents = aggregate.get("n_documents", 0)
    if config.loss_evals:
        eval_wrapped = CacheAndModel(cache, model)
        eval_metrics = {}
        for ds_config in config.loss_evals:
            metrics = evaluate_perplexity(
                config=config,
                model=eval_wrapped,
                cache=cache,
                eval_dataset=ds_config.dataset.instantiate(
                    tokenizer=tokenizer, seed=config.seed
                ),
                ds_config=ds_config,
                optimizer_step=n_documents or 1,
                epoch=0,
                local_rank=device,
                cache_tuning=True,
            )
            if metrics:
                eval_metrics[metrics["name"]] = metrics
        aggregate["eval_metrics"] = eval_metrics

    value_norms = _value_norm_summary(cache)
    aggregate["value_norms"] = value_norms

    # Lightweight summary for the free-GPU A/B launcher.
    (run_dir / "phase2_summary.json").write_text(
        json.dumps(
            {
                "run_name": config.name,
                "run_dir": str(run_dir),
                "n_documents": n_documents,
                "wall_clock_s": wall_clock,
                "mean_mse_last_doc": (
                    aggregate["per_document"][-1].get("mean_mse")
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
                    "ridge_scale": objective_cfg.ridge_scale,
                    "ridge_lambda": objective_cfg.ridge_lambda,
                    "ridge_lambda_min": objective_cfg.ridge_lambda_min,
                    "delta_weight": objective_cfg.delta_weight,
                },
            },
            indent=2,
        )
    )

    save_cache(config, cache, optimizer_step=n_documents or 1)
    wrapped.remove_hooks()
    logger.info(
        "Per-document AM Phase 2 complete in %.1fs (%d documents). Saved to %s",
        wall_clock,
        n_documents,
        run_dir,
    )
    return aggregate
