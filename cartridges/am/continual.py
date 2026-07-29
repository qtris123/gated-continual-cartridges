"""Per-document AM Phase 2 orchestration (teacher [cartridge || ICL(doc)])."""

from __future__ import annotations

import hashlib
import time
from logging import getLogger
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cartridges.am.finetune import (
    AttentionMatchingFinetuningConfig,
    _collect_reference_queries,
    apply_document_am_write_to_cache,
)
from cartridges.am.query_accum import AMQueryAccumulator
from cartridges.am.ranking import rank_am_slots
from cartridges.am.reference_data import (
    build_reference_dataloader,
    canonical_document_prompt,
    cleanup_reference_parquet,
    group_conversations_by_document,
    limit_conversations,
    load_conversations,
    load_old_reference_bank,
)
from cartridges.am.teacher import compute_teacher_targets, prefill_document_kv_cache
from cartridges.sparse_cache_finetuning import CacheTFIDFRanker

logger = getLogger(__name__)


def _doc_slug(system_prompt: str, index: int) -> str:
    digest = hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:8]
    return f"doc-{index:03d}-{digest}"


def run_per_document_am_phase2(
    cache: nn.Module,
    wrapped_model: nn.Module,
    model: nn.Module,
    tokenizer,
    mt_data_path: str,
    tfidf_ranker: CacheTFIDFRanker,
    config: AttentionMatchingFinetuningConfig,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    local_rank: torch.device,
    run_dir: Optional[Path] = None,
    save_cache_fn=None,
) -> dict:
    """Run one closed-form AM write per unique document (system_prompt group)."""
    was_training = wrapped_model.training
    wrapped_model.eval()

    conversations = load_conversations(mt_data_path)
    doc_groups = group_conversations_by_document(conversations)
    logger.info("Per-document AM: %d unique documents in %s", len(doc_groups), mt_data_path)

    old_query_acc: Optional[AMQueryAccumulator] = None
    old_target_bank: Optional[dict[tuple[int, int], torch.Tensor]] = None
    old_batch_count = 0
    if config.enable_old_reference_guard and config.old_ref_data_path:
        old_conversations = load_old_reference_bank(
            config.old_ref_data_path,
            max_examples=config.old_ref_max_examples,
        )
        old_loader, old_tmp = build_reference_dataloader(
            old_conversations,
            tokenizer,
            seed=0,
        )
        try:
            old_query_acc, _, old_batch_count = _collect_reference_queries(
                wrapped_model,
                cache,
                old_loader,
                config,
                n_layers,
                n_kv_heads,
                head_dim,
                local_rank,
                max_batches=config.decoupled_ref_batches,
            )
        finally:
            cleanup_reference_parquet(old_tmp)

        old_target_bank = {}
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
                ).to(device=local_rank, dtype=layer_keys.dtype)
                head_beta = (
                    layer_beta[0, head_idx]
                    if layer_beta is not None
                    else None
                )
                old_target_bank[(layer_idx, head_idx)] = compute_teacher_targets(
                    old_queries,
                    layer_keys[0, head_idx],
                    layer_values[0, head_idx],
                    head_dim,
                    attention_bias=head_beta,
                ).detach().cpu()

    per_doc_stats = []
    t_total = time.time()

    for doc_idx, (document_id, doc_conversations) in enumerate(doc_groups.items()):
        system_prompt = canonical_document_prompt(doc_conversations)
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

        limited = limit_conversations(
            doc_conversations,
            config.max_ref_examples_per_doc,
            seed=doc_idx,
        )
        doc_loader, doc_tmp = build_reference_dataloader(
            limited,
            tokenizer,
            seed=doc_idx,
        )

        t_doc = time.time()
        try:
            t_prefill = time.time()
            doc_kv = prefill_document_kv_cache(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                attn_config=cache.config,
                device=local_rank,
                cartridge_cache=cache,
            )
            prefill_s = time.time() - t_prefill

            query_acc, _, batch_count = _collect_reference_queries(
                wrapped_model,
                cache,
                doc_loader,
                config,
                n_layers,
                n_kv_heads,
                head_dim,
                local_rank,
                max_batches=len(doc_loader),
            )

            access_scores = query_acc.get_access_scores()
            old_access_scores = (
                old_query_acc.get_access_scores()
                if old_query_acc is not None and config.enable_old_reference_guard
                else None
            )
            grad_mask, ranking_info = rank_am_slots(
                access_scores,
                config.top_t,
                tfidf_ranker,
                config,
                old_access_scores=old_access_scores,
                step=doc_idx + 1,
            )

            # LIT-006 / MECH-006: on-policy, layer-sequential re-extraction. The
            # kwarg is passed ONLY when the knob is on, so a stock run's call
            # signature is byte-identical (RUNBOOK 6.10 / MECH-000: an
            # unconditional new kwarg once crashed every AM run through the
            # sibling-`cartridges` import path).
            onpolicy_kwargs: dict = {}
            onpolicy_group = int(getattr(config, "onpolicy_layers", 0) or 0)
            if onpolicy_group > 0:
                refresh_doc_kv = bool(
                    getattr(config, "onpolicy_refresh_doc_kv", False)
                )

                def _onpolicy_refresh_fn(layer_idx: int, group_size: int):
                    """Re-extract reference queries from the UPDATED cartridge.

                    Layers `< layer_idx` have already been written, so this
                    forward pass sees the perturbed residual stream and produces
                    the queries the remaining layers will actually be asked.
                    """
                    fresh_acc, _, _ = _collect_reference_queries(
                        wrapped_model,
                        cache,
                        doc_loader,
                        config,
                        n_layers,
                        n_kv_heads,
                        head_dim,
                        local_rank,
                        max_batches=len(doc_loader),
                    )
                    fresh_kv = None
                    if refresh_doc_kv:
                        fresh_kv = prefill_document_kv_cache(
                            model=model,
                            tokenizer=tokenizer,
                            system_prompt=system_prompt,
                            attn_config=cache.config,
                            device=local_rank,
                            cartridge_cache=cache,
                        )
                    return fresh_acc, fresh_kv

                onpolicy_kwargs["onpolicy_refresh_fn"] = _onpolicy_refresh_fn

            am_stats = apply_document_am_write_to_cache(
                cache=cache,
                mask=grad_mask,
                query_accumulator=query_acc,
                doc_kv=doc_kv,
                config=config,
                n_layers=n_layers,
                head_dim=head_dim,
                old_query_accumulator=old_query_acc,
                old_target_bank=old_target_bank,
                **onpolicy_kwargs,
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
                "extra": dict(getattr(am_stats, "extra", None) or {}),
            }
            per_doc_stats.append(doc_record)

            if run_dir is not None:
                stats_path = run_dir / f"am_doc_{slug}.pt"
                torch.save(
                    {
                        "doc_record": doc_record,
                        "ranking_info": ranking_info,
                        "am_stats": am_stats,
                    },
                    stats_path,
                )

            if config.save_after_each_document and save_cache_fn is not None:
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
    logger.info(
        "Per-document AM Phase 2 complete: %d documents in %.1fs",
        len(per_doc_stats),
        aggregate["wall_clock_s"],
    )
    return aggregate
