"""Slot ranking strategies (TF-IDF / attention-mass / residual-budget)."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Optional

import torch

from cartridges.sparse_cache_finetuning import (
    CacheTFIDFRanker,
    GradientMask,
    TFIDFRankingInfo,
)

if TYPE_CHECKING:
    from cartridges.am.finetune import AttentionMatchingFinetuningConfig

logger = getLogger(__name__)


def _rank_attention_mass_per_layer(
    access_scores: torch.Tensor,
    top_t: int,
) -> tuple[GradientMask, TFIDFRankingInfo]:
    scores = access_scores.float().cpu()
    n_layers, n_tokens = scores.shape
    k = min(top_t, n_tokens)
    totals = scores.sum(-1, keepdim=True).clamp(min=1e-12)
    tf = scores / totals
    _, top = torch.topk(tf, k=k, dim=-1, largest=True)
    mask = GradientMask(
        granularity="per_layer",
        positions_per_layer={l: top[l] for l in range(n_layers)},
        n_tokens=n_tokens,
        top_t=k,
    )
    info = TFIDFRankingInfo(step=-1, tf=tf, tfidf=tf, mask=mask)
    return mask, info


def _rank_residual_budget_per_layer(
    access_scores: torch.Tensor,
    top_t: int,
    old_access_scores: Optional[torch.Tensor] = None,
    idf_scores: Optional[torch.Tensor] = None,
    idf_prior_weight: float = 0.0,
    min_top_t_per_layer: int = 1,
) -> tuple[GradientMask, TFIDFRankingInfo]:
    """Allocate layer budgets from access pressure and old/new conflict.

    This is the first AM-native budget proxy. Until a non-self target supplies
    true AM residuals, access conflict is used as a cheap residual-pressure
    surrogate.
    """
    scores = access_scores.float().cpu()
    n_layers, n_tokens = scores.shape
    totals = scores.sum(-1, keepdim=True).clamp(min=1e-12)
    tf = scores / totals

    if old_access_scores is not None:
        old = old_access_scores.float().cpu()
        old_tf = old / old.sum(-1, keepdim=True).clamp(min=1e-12)
        conflict = (tf - old_tf).abs().sum(-1)
        layer_pressure = conflict.clamp(min=1e-6)
    else:
        layer_pressure = tf.std(-1).clamp(min=1e-6)

    total_budget = min(top_t * n_layers, n_tokens * n_layers)
    raw_budget = layer_pressure / layer_pressure.sum() * total_budget
    budgets = raw_budget.round().long().clamp(min=min_top_t_per_layer, max=n_tokens)

    # Keep total budget near top_t * n_layers after rounding/clamping.
    while budgets.sum().item() > total_budget:
        idx = torch.argmax(budgets.float()).item()
        if budgets[idx] <= min_top_t_per_layer:
            break
        budgets[idx] -= 1
    while budgets.sum().item() < total_budget:
        idx = torch.argmax(layer_pressure).item()
        if budgets[idx] >= n_tokens:
            break
        budgets[idx] += 1

    score = tf
    if idf_scores is not None and idf_prior_weight > 0:
        idf = idf_scores.float().cpu()
        score = score + idf_prior_weight * (idf / idf.max().clamp(min=1e-12))

    positions = {}
    for layer_idx in range(n_layers):
        k = int(min(max(budgets[layer_idx].item(), 1), n_tokens))
        _, top = torch.topk(score[layer_idx], k=k, largest=True)
        positions[layer_idx] = top

    mask = GradientMask(
        granularity="per_layer",
        positions_per_layer=positions,
        n_tokens=n_tokens,
        top_t=top_t,
    )
    info = TFIDFRankingInfo(step=-1, tf=tf, tfidf=score, mask=mask)
    return mask, info


def rank_am_slots(
    access_scores: torch.Tensor,
    top_t: int,
    tfidf_ranker: CacheTFIDFRanker,
    config: "AttentionMatchingFinetuningConfig",
    old_access_scores: Optional[torch.Tensor] = None,
    step: int = 0,
) -> tuple[GradientMask, TFIDFRankingInfo]:
    if config.slot_selection == "tfidf":
        return tfidf_ranker.rank_positions(
            access_scores,
            top_t=top_t,
            step=step,
            return_info=True,
        )

    if config.granularity != "per_layer":
        logger.warning(
            "%s selection currently supports per_layer only; falling back to TF-IDF",
            config.slot_selection,
        )
        return tfidf_ranker.rank_positions(
            access_scores,
            top_t=top_t,
            step=step,
            return_info=True,
        )

    if config.slot_selection == "attention_mass":
        mask, info = _rank_attention_mass_per_layer(access_scores, top_t)
    else:
        mask, info = _rank_residual_budget_per_layer(
            access_scores,
            top_t,
            old_access_scores=old_access_scores,
            idf_scores=tfidf_ranker.get_idf_scores(),
            idf_prior_weight=config.idf_prior_weight,
            min_top_t_per_layer=config.min_top_t_per_layer,
        )
    info.step = step
    return mask, info
