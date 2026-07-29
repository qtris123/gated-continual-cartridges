"""AM finetuning config, cache-update application, and decoupled Phase-2 solve.

Combines TF-IDF sparse slot selection (Lin et al., 2025) with closed-form
Attention Matching value updates (Zweiger et al., 2026) instead of gradient descent.

Execution modes (Phase 2):
- ``per_document``: one closed-form AM write per unique ``system_prompt`` (default)
- ``legacy_decoupled``: single global TF-IDF solve over shuffled MT batches
- ``train_loop``: legacy optimizer-step-shaped loop in ``train.py`` (deprecated)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
import time
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from pydrantic import BaseConfig

from cartridges.am.key_select import refit_beta_nnls, rewrite_keys_on_support
from cartridges.am.query_accum import (
    AMQueryAccumulator,
    AMTargetAccumulator,
    install_teacher_attention_capture_hooks,
)
from cartridges.am.ranking import rank_am_slots
from cartridges.am.teacher import (
    compute_teacher_log_mass,
    compute_teacher_targets,
    concat_teacher_kv,
)
from cartridges.am.value_solve import (
    guarded_sparse_am_value_update,
    oracle_teacher_value_write,
    sparse_am_value_update,
)
from cartridges.sparse_cache_finetuning import (
    CacheTFIDFRanker,
    GradientMask,
    TFIDFRankingInfo,
    install_query_capture_hooks,
)

logger = getLogger(__name__)


class AttentionMatchingFinetuningConfig(BaseConfig):
    enabled: bool = False

    # TF-IDF slot selection (shared with sparse gradient finetuning)
    top_t: int = 64
    background_indices_path: Optional[str] = None
    num_background_batches: int = 1000
    use_idf: bool = True
    idf_smoothing: float = 1.0
    collect_background_stats: bool = False
    background_top_k_per_batch: int = 128
    granularity: Literal["global", "per_layer", "per_head"] = "per_layer"
    freeze_keys: bool = True

    # Phase 2 execution / target
    execution_mode: Literal["per_document", "legacy_decoupled", "train_loop"] = "per_document"
    target_mode: Literal["cartridge_plus_doc", "self", "teacher_attention"] = "cartridge_plus_doc"
    enable_old_reference_guard: bool = False
    old_ref_data_path: Optional[str] = None
    old_ref_max_examples: int = 64
    key_mode: Literal["freeze", "highest_attention", "omp"] = "freeze"
    enable_beta: Optional[bool] = None
    beta_fit_scope: Literal["all", "selected"] = "selected"
    # B-SOLVE / LIT-002 (AM paper App. C.2 "Stabilizing beta", Algorithm 3).
    # All three default to the historical behaviour, so a run that does not set
    # them is bit-identical.
    #   beta_box    -> symmetric box on the fitted log-weights (paper: 3.0 for
    #                  highest-attention keys). None = unbounded above, floored
    #                  at log(1e-12) = -27.6.
    #   nnls_iters  -> projected-gradient steps (paper: 2). Historical: 200.
    #   nnls_driver -> lstsq driver for the NNLS warm start. None keeps torch's
    #                  CUDA default `gels`, which returns NaN for rank-deficient
    #                  input WITHOUT raising; 'gelsd' is rank-revealing.
    #   beta_target -> 'residual' subtracts the non-selected slots' mass from the
    #                  teacher target (this tree's own invention); 'full' fits the
    #                  selected keys against the full teacher mass, as the paper
    #                  does over all compacted keys.
    beta_box: Optional[float] = None
    nnls_iters: int = 200
    nnls_driver: Optional[str] = None
    beta_target: Literal["residual", "full"] = "residual"
    max_ref_examples_per_doc: int = 32
    save_after_each_document: bool = True

    # AM-specific (legacy + shared)
    queries_per_batch: Literal["last_token", "all_tokens"] = "all_tokens"
    ridge_lambda: float = 1e-4
    ridge_scale: Literal["spectral", "frobenius", "fixed", "absolute"] = "spectral"
    ridge_lambda_min: float = 0.0
    update_interval: int = 1
    max_queries_per_head: int = 64
    max_am_steps: int = -1  # cap optimizer steps; -1 = use epochs * dataloader
    decoupled_ref_batches: int = 5
    compute_update_stats: bool = True
    old_reference_weight: float = 1.0
    delta_weight: float = 0.0
    slot_selection: Literal["tfidf", "attention_mass", "residual_budget"] = "tfidf"
    idf_prior_weight: float = 0.0
    min_top_t_per_layer: int = 1

    # B-ROUTE write-ceiling oracle (OPT-IN, default off -> bit-identical stock runs).
    # When on, the per-document write skips the closed-form solve and copies the
    # teacher's own document value vectors into the selected slots instead.
    oracle_write: bool = False
    oracle_write_assign: Literal["mass_ranked", "sequential"] = "mass_ranked"

    # B-ROPE: rotary base used when the teacher path re-positions the reference
    # queries by `doc_rope_offset = T_doc` before scoring them against the document
    # keys (`core._apply_rope_offset_to_queries`). The AM package hard-coded
    # 10000.0 everywhere, but the model's own RoPE comes from its HF config
    # (Qwen3-4B-Instruct-2507 -> 5e6). Default stays 10000.0 so every stock run is
    # bit-identical; set `AM_ROPE_THETA` to opt in to the model's true base.
    rope_theta: float = 10000.0

    # B-ROPE hazard H2 / LIT-026: when `key_mode != "freeze"`, a *document* key that
    # wins selection is installed into a cartridge slot with NO counter-rotation --
    # it was scored against a query rotated forward by `doc_rope_offset = T_doc`, but
    # the student/eval scores it with the raw query, so the installed key is off by a
    # multi-thousand-position phase. `key_reposition=True` re-bases every doc-sourced
    # row by `-doc_rope_offset` (`phase1._rope_reposition`, the operator
    # `initial_am_compaction` already uses) so the logit that selected the key is the
    # logit it delivers. Default False -> `key_mode="freeze"` runs are untouched and
    # a key run reproduces the historical (phase-corrupted) behaviour exactly.
    key_reposition: bool = False


@dataclass
class AMUpdateStats:
    step: int
    mse_per_layer: Dict[int, float]
    mean_mse: float
    n_queries: int
    mask: Optional[GradientMask] = None
    timing: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def apply_am_update_to_cache(
    cache: nn.Module,
    mask: GradientMask,
    query_accumulator: AMQueryAccumulator,
    n_layers: int,
    head_dim: int,
    target_mode: str = "self",
    ridge_lambda: float = 1e-4,
    freeze_keys: bool = True,
    max_queries_per_head: int = 64,
    compute_stats: bool = True,
    old_query_accumulator: Optional[AMQueryAccumulator] = None,
    target_accumulator: Optional[AMTargetAccumulator] = None,
    old_reference_weight: float = 0.0,
    delta_weight: float = 0.0,
    ridge_scale: str = "spectral",
    ridge_lambda_min: float = 0.0,
) -> AMUpdateStats:
    """Apply sparse AM value updates to cache using TF-IDF mask.

    For each (layer, head), solves residual lstsq on top-t value positions.
    """
    device = cache.trainable_values[0].device
    mse_per_layer: dict[int, float] = {}
    total_mse = 0.0
    n_updated = 0
    total_queries = 0

    for layer_idx in range(n_layers):
        v_param = cache.trainable_values[layer_idx]
        k_param = cache.trainable_keys[layer_idx]
        n_kv_heads = v_param.shape[1]
        n_tokens = v_param.shape[2]

        layer_positions = mask.positions_per_layer.get(layer_idx)
        if layer_positions is None:
            continue

        layer_mses = []

        if not query_accumulator._queries[layer_idx]:
            continue

        q_batch = query_accumulator._queries[layer_idx][0]
        n_q_heads_actual = q_batch.shape[1]

        for head_idx in range(n_kv_heads):
            if mask.granularity == "per_head":
                selected = layer_positions[head_idx].to(device)
            else:
                selected = layer_positions.to(device)

            keys = k_param[0, head_idx].detach()  # (T, d)
            values = v_param[0, head_idx].detach()  # (T, d)

            queries = query_accumulator.get_layer_head_queries(
                layer_idx, head_idx,
                n_q_heads=n_q_heads_actual,
                n_kv_heads=n_kv_heads,
            )

            if queries.numel() == 0:
                continue

            queries = queries.to(device=device, dtype=keys.dtype)

            if target_mode == "self":
                targets = None
            elif target_mode == "teacher_attention":
                if target_accumulator is None:
                    raise ValueError("target_mode='teacher_attention' requires target_accumulator")
                targets = target_accumulator.get_layer_head_targets(
                    layer_idx, head_idx,
                    n_q_heads=n_q_heads_actual,
                    n_kv_heads=n_kv_heads,
                )
                if targets.numel() == 0:
                    continue
                if targets.shape[0] != queries.shape[0]:
                    n = min(targets.shape[0], queries.shape[0])
                    queries = queries[:n]
                    targets = targets[:n]
                targets = targets.to(device=device, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported AM target_mode: {target_mode}")

            if queries.shape[0] > max_queries_per_head:
                idx = torch.randperm(queries.shape[0], device=device)[:max_queries_per_head]
                queries = queries[idx]
                if targets is not None:
                    targets = targets[idx]

            total_queries += queries.shape[0]

            old_queries = None
            if old_query_accumulator is not None and old_query_accumulator._queries[layer_idx]:
                old_queries = old_query_accumulator.get_layer_head_queries(
                    layer_idx, head_idx,
                    n_q_heads=n_q_heads_actual,
                    n_kv_heads=n_kv_heads,
                )
                if old_queries.numel() > 0:
                    old_queries = old_queries.to(device=device, dtype=keys.dtype)
                    if old_queries.shape[0] > max_queries_per_head:
                        idx = torch.randperm(old_queries.shape[0], device=device)[:max_queries_per_head]
                        old_queries = old_queries[idx]

            if old_queries is not None or delta_weight > 0:
                new_values, stats = guarded_sparse_am_value_update(
                    keys, values, queries, selected,
                    old_queries=old_queries,
                    targets_new=targets,
                    targets_old=None,
                    head_dim=head_dim,
                    ridge_lambda=ridge_lambda,
                    ridge_scale=ridge_scale,
                    ridge_lambda_min=ridge_lambda_min,
                    old_reference_weight=old_reference_weight,
                    delta_weight=delta_weight,
                    compute_stats=compute_stats,
                )
                mse = stats["mse_new"]
            else:
                new_values, stats = sparse_am_value_update(
                    keys, values, queries, selected,
                    targets=targets, head_dim=head_dim,
                    ridge_lambda=ridge_lambda,
                    ridge_scale=ridge_scale,
                    ridge_lambda_min=ridge_lambda_min,
                    compute_stats=compute_stats,
                )
                mse = stats["mse"]

            with torch.no_grad():
                v_param[0, head_idx].copy_(new_values.to(v_param.dtype))

            if mse is not None:
                layer_mses.append(mse)
            n_updated += 1

        if layer_mses:
            mse_per_layer[layer_idx] = sum(layer_mses) / len(layer_mses)
            total_mse += mse_per_layer[layer_idx]

    mean_mse = total_mse / max(len(mse_per_layer), 1) if mse_per_layer else 0.0
    return AMUpdateStats(
        step=-1,
        mse_per_layer=mse_per_layer,
        mean_mse=mean_mse,
        n_queries=total_queries,
        mask=mask,
    )


def _should_fit_beta(config: AttentionMatchingFinetuningConfig) -> bool:
    """Whether to fit per-key log-bias beta. Decoupled from `key_mode` (B-SOLVE).

    Historically the unset fallback was `config.key_mode != "freeze"`, which
    (a) silently switched the beta/NNLS path ON for every key experiment and
    (b) made "beta with frozen keys" -- the configuration B-ROUTE actually needs
    -- expressible only by setting `ENABLE_BETA=1` explicitly. beta is now an
    independent axis: unset == off, whatever `key_mode` is. Every run in
    `state/results.csv` used `key_mode="freeze"`, where both rules agree, so no
    historical configuration changes behaviour.
    """
    if config.enable_beta is True:
        return True
    if config.enable_beta is False:
        return False
    if config.key_mode != "freeze":
        logger.warning(
            "ENABLE_BETA is unset with key_mode=%s: beta is OFF (it is now an "
            "independent axis; set ENABLE_BETA=1 to fit it).",
            config.key_mode,
        )
    return False


def _collect_reference_queries(
    wrapped_model: nn.Module,
    cache: nn.Module,
    source_dataloader,
    config: AttentionMatchingFinetuningConfig,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    local_rank: torch.device,
    max_batches: Optional[int] = None,
    collect_teacher_targets: bool = False,
) -> tuple[AMQueryAccumulator, Optional[AMTargetAccumulator], int]:
    """Collect reference queries (and optional legacy teacher targets) from a dataloader."""
    query_acc = AMQueryAccumulator(
        granularity=config.granularity,
        queries_per_batch=config.queries_per_batch,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        device=local_rank,
    )
    target_acc = (
        AMTargetAccumulator(n_layers=n_layers, n_kv_heads=n_kv_heads)
        if collect_teacher_targets
        else None
    )
    batch_limit = max_batches if max_batches is not None else config.decoupled_ref_batches
    batch_count = 0
    with torch.no_grad():
        for batch in source_dataloader:
            if batch_count >= batch_limit:
                break

            input_ids = batch.input_ids.to(local_rank)
            seq_ids = batch.element_ids.to(local_rank)
            position_ids = batch.position_ids.to(local_rank)

            wrapped_model(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
            )
            captured_q = wrapped_model.get_captured_queries()
            if captured_q:
                query_acc.accumulate_from_hooks(
                    captured_q,
                    cache,
                    scaling=head_dim ** -0.5,
                    seq_ids=seq_ids,
                )

            if target_acc is not None:
                teacher_captured, teacher_handles = install_teacher_attention_capture_hooks(
                    wrapped_model.model
                )
                try:
                    wrapped_model.model(
                        input_ids=input_ids,
                        seq_ids=seq_ids,
                        position_ids=position_ids,
                        use_cache=False,
                        past_key_values=None,
                    )
                    if teacher_captured:
                        target_acc.accumulate_from_hooks(teacher_captured)
                finally:
                    for handle in teacher_handles:
                        handle.remove()

            batch_count += 1

    return query_acc, target_acc, batch_count


def apply_document_am_write_to_cache(
    cache: nn.Module,
    mask: GradientMask,
    query_accumulator: AMQueryAccumulator,
    doc_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    config: AttentionMatchingFinetuningConfig,
    n_layers: int,
    head_dim: int,
    old_query_accumulator: Optional[AMQueryAccumulator] = None,
    old_target_bank: Optional[dict[tuple[int, int], torch.Tensor]] = None,
) -> AMUpdateStats:
    """Apply one per-document AM write with teacher [cartridge || doc KV]."""
    device = cache.trainable_values[0].device
    mse_per_layer: dict[int, float] = {}
    total_mse = 0.0
    total_queries = 0
    use_old_guard = (
        config.enable_old_reference_guard
        and old_query_accumulator is not None
    )
    fit_beta = _should_fit_beta(config)
    # B-SOLVE / LIT-002 beta-fit knobs. `getattr` defaults reproduce the historical
    # behaviour exactly, so an older config object (or an unset flag) is unchanged.
    beta_kwargs = {
        "n_iters": int(getattr(config, "nnls_iters", 200)),
        "beta_box": getattr(config, "beta_box", None),
        "nnls_driver": getattr(config, "nnls_driver", None),
        "target_mode": getattr(config, "beta_target", "residual"),
    }
    beta_fit_info: list[dict] = []
    beta_per_layer: dict[int, list[torch.Tensor]] = {}
    # B-ROPE: 10000.0 is the historical hard-coded AM default; `getattr` keeps this
    # working against an older config object that has no such field.
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    # B-ROPE H2: opt-in counter-rotation of doc-sourced keys on install (MECH-005).
    key_reposition = bool(getattr(config, "key_reposition", False))
    key_rewrite_info: list[dict] = []
    oracle_write = bool(getattr(config, "oracle_write", False))
    oracle_mass_on_S: dict[int, list[float]] = {}
    oracle_doc_mass: dict[int, list[float]] = {}
    oracle_n_written: list[int] = []
    # B-CASCADE diagnostics (read-only; never touch the solution).
    solve_mass_on_S: dict[int, list[float]] = {}
    solve_v_absmax: dict[int, list[float]] = {}
    n_queries_available: list[int] = []
    n_queries_used: list[int] = []

    for layer_idx in range(n_layers):
        v_param = cache.trainable_values[layer_idx]
        k_param = cache.trainable_keys[layer_idx]
        beta_param = (
            cache.trainable_beta[layer_idx]
            if getattr(cache, "trainable_beta", None) is not None
            else None
        )
        n_kv_heads = v_param.shape[1]

        layer_positions = mask.positions_per_layer.get(layer_idx)
        if layer_positions is None or not query_accumulator._queries[layer_idx]:
            continue

        k_doc_layer, v_doc_layer = doc_kv[layer_idx]
        layer_mses = []
        q_batch = query_accumulator._queries[layer_idx][0]
        n_q_heads_actual = q_batch.shape[1]

        for head_idx in range(n_kv_heads):
            if mask.granularity == "per_head":
                selected = layer_positions[head_idx].to(device)
            else:
                selected = layer_positions.to(device)

            trainable_keys = k_param[0, head_idx].detach().clone()
            trainable_values = v_param[0, head_idx].detach().clone()
            n_frozen = getattr(cache, "_num_frozen_tokens", 0)
            if n_frozen > 0:
                frozen_keys = cache.frozen_keys[layer_idx][0, head_idx].detach()
                frozen_values = cache.frozen_values[layer_idx][0, head_idx].detach()
                keys = torch.cat([frozen_keys, trainable_keys], dim=0)
                values = torch.cat([frozen_values, trainable_values], dim=0)
                selected_full = selected + n_frozen
            else:
                keys = trainable_keys
                values = trainable_values
                selected_full = selected
            original_keys = keys.clone()
            trainable_beta = (
                beta_param[0, head_idx].detach().to(torch.float32).clone()
                if beta_param is not None
                else torch.zeros(trainable_keys.shape[0], device=device, dtype=torch.float32)
            )
            base_beta = (
                torch.cat(
                    [
                        torch.zeros(n_frozen, device=device, dtype=torch.float32),
                        trainable_beta,
                    ]
                )
                if n_frozen > 0
                else trainable_beta
            )
            k_doc = k_doc_layer[head_idx].to(device=device, dtype=keys.dtype)
            v_doc = v_doc_layer[head_idx].to(device=device, dtype=values.dtype)

            queries = query_accumulator.get_layer_head_queries(
                layer_idx, head_idx,
                n_q_heads=n_q_heads_actual,
                n_kv_heads=n_kv_heads,
            )
            if queries.numel() == 0:
                continue
            queries = queries.to(device=device, dtype=keys.dtype)

            # How many reference queries the accumulator could actually supply for
            # this (layer, head) BEFORE the subsample — this is the hard ceiling on
            # `max_queries_per_head` (B-CASCADE). Recorded, never padded.
            n_queries_available.append(int(queries.shape[0]))

            if queries.shape[0] > config.max_queries_per_head:
                idx = torch.randperm(queries.shape[0], device=device)[: config.max_queries_per_head]
                queries = queries[idx]

            n_queries_used.append(int(queries.shape[0]))
            total_queries += queries.shape[0]

            old_queries = None
            targets_old = None
            if use_old_guard and old_query_accumulator._queries[layer_idx]:
                old_queries = old_query_accumulator.get_layer_head_queries(
                    layer_idx, head_idx,
                    n_q_heads=n_q_heads_actual,
                    n_kv_heads=n_kv_heads,
                )
                if old_queries.numel() > 0:
                    old_queries = old_queries.to(device=device, dtype=keys.dtype)
                    if old_target_bank is not None:
                        targets_old = old_target_bank[(layer_idx, head_idx)].to(
                            device=device,
                            dtype=torch.float32,
                        )
                    if old_queries.shape[0] > config.max_queries_per_head:
                        idx = torch.randperm(old_queries.shape[0], device=device)[
                            : config.max_queries_per_head
                        ]
                        old_queries = old_queries[idx]
                        if targets_old is not None:
                            targets_old = targets_old[idx]
                    # Preserve the pre-write cartridge behavior, including its
                    # current routing and beta, when keys or beta are rewritten.
                    if targets_old is None:
                        targets_old = compute_teacher_targets(
                            old_queries,
                            original_keys,
                            values,
                            head_dim,
                            attention_bias=base_beta,
                        )

            k_teacher, v_teacher = concat_teacher_kv(keys, values, k_doc, v_doc)
            teacher_bias = torch.cat(
                [
                    base_beta,
                    torch.zeros(k_doc.shape[0], device=device, dtype=torch.float32),
                ]
            )
            n_cartridge_keys = keys.shape[0]
            doc_rope_offset = k_doc.shape[0]
            teacher_rope_kwargs = {
                "n_cartridge_keys": n_cartridge_keys,
                "doc_rope_offset": doc_rope_offset,
                "rope_theta": rope_theta,
            }
            targets = compute_teacher_targets(
                queries,
                k_teacher,
                v_teacher,
                head_dim,
                attention_bias=teacher_bias,
                **teacher_rope_kwargs,
            )
            target_log_mass = compute_teacher_log_mass(
                queries,
                k_teacher,
                head_dim,
                attention_bias=teacher_bias,
                **teacher_rope_kwargs,
            )

            if config.key_mode != "freeze":
                # Non-selected cartridge keys remain fixed. Compact only the
                # replaceable old support plus the new document into S.
                candidate_keys = torch.cat(
                    [original_keys[selected_full], k_doc],
                    dim=0,
                )
                rewrite_info: dict = {"layer": layer_idx, "head": head_idx}
                keys = rewrite_keys_on_support(
                    keys,
                    selected_full,
                    candidate_keys,
                    queries,
                    mode=config.key_mode,
                    head_dim=head_dim,
                    doc_key_start=selected_full.numel(),
                    doc_rope_offset=doc_rope_offset,
                    rope_theta=rope_theta,
                    reposition=key_reposition,
                    info=rewrite_info,
                )
                key_rewrite_info.append(rewrite_info)

            head_beta = base_beta
            if fit_beta and beta_param is not None:
                fit_info: dict = {"layer": layer_idx, "head": head_idx}
                if config.beta_fit_scope == "selected":
                    fit_idx = selected_full
                else:
                    fit_idx = torch.arange(
                        n_frozen,
                        keys.shape[0],
                        device=device,
                        dtype=torch.long,
                    )
                beta_full = refit_beta_nnls(
                    keys,
                    queries,
                    target_log_mass,
                    head_dim,
                    selected_indices=fit_idx,
                    base_beta=base_beta,
                    info=fit_info,
                    **beta_kwargs,
                )
                if not torch.isfinite(beta_full).all():
                    # Fail loudly: a NaN beta would otherwise be written into the
                    # cache and poison the value solve (EXP-005b/EXP-006).
                    raise RuntimeError(
                        f"refit_beta_nnls returned non-finite beta at layer "
                        f"{layer_idx} head {head_idx}: "
                        f"{int((~torch.isfinite(beta_full)).sum().item())} of "
                        f"{beta_full.numel()} entries. fit_info={fit_info}"
                    )
                beta_fit_info.append(fit_info)
                beta_per_layer.setdefault(layer_idx, []).append(
                    beta_full[fit_idx].detach().float().cpu()
                )
                with torch.no_grad():
                    beta_param[0, head_idx].copy_(
                        beta_full[n_frozen:].to(beta_param.dtype)
                    )
                if hasattr(cache, "enable_attention_bias"):
                    cache.enable_attention_bias(True)
                head_beta = beta_full

            if oracle_write:
                # B-ROUTE write-ceiling oracle: skip the solve entirely and put the
                # teacher's own document values into the selected slots.
                new_values, stats = oracle_teacher_value_write(
                    keys, values, queries, selected_full,
                    v_doc=v_doc,
                    k_teacher=k_teacher,
                    targets=targets,
                    head_dim=head_dim,
                    attention_bias=head_beta,
                    teacher_bias=teacher_bias,
                    n_cartridge_keys=n_cartridge_keys,
                    doc_rope_offset=doc_rope_offset,
                    rope_theta=rope_theta,
                    assign=getattr(config, "oracle_write_assign", "mass_ranked"),
                    compute_stats=config.compute_update_stats,
                )
                mse = stats["mse"]
                oracle_mass_on_S.setdefault(layer_idx, []).append(stats["mass_on_S_mean"])
                if stats.get("teacher_doc_mass_mean") is not None:
                    oracle_doc_mass.setdefault(layer_idx, []).append(
                        stats["teacher_doc_mass_mean"]
                    )
                oracle_n_written.append(stats["n_written"])
            elif use_old_guard or config.delta_weight > 0:
                new_values, stats = guarded_sparse_am_value_update(
                    keys, values, queries, selected_full,
                    old_queries=old_queries,
                    targets_new=targets,
                    targets_old=targets_old,
                    head_dim=head_dim,
                    ridge_lambda=config.ridge_lambda,
                    ridge_scale=config.ridge_scale,
                    ridge_lambda_min=config.ridge_lambda_min,
                    old_reference_weight=config.old_reference_weight if use_old_guard else 0.0,
                    delta_weight=config.delta_weight,
                    compute_stats=config.compute_update_stats,
                    attention_bias=head_beta,
                )
                mse = stats["mse_new"]
                if stats.get("mass_on_S_mean") is not None:
                    solve_mass_on_S.setdefault(layer_idx, []).append(stats["mass_on_S_mean"])
                if stats.get("v_selected_absmax_after") is not None:
                    solve_v_absmax.setdefault(layer_idx, []).append(
                        stats["v_selected_absmax_after"]
                    )
            else:
                new_values, stats = sparse_am_value_update(
                    keys, values, queries, selected_full,
                    targets=targets,
                    head_dim=head_dim,
                    ridge_lambda=config.ridge_lambda,
                    ridge_scale=config.ridge_scale,
                    ridge_lambda_min=config.ridge_lambda_min,
                    compute_stats=config.compute_update_stats,
                    attention_bias=head_beta,
                )
                mse = stats["mse"]
                if stats.get("mass_on_S_mean") is not None:
                    solve_mass_on_S.setdefault(layer_idx, []).append(stats["mass_on_S_mean"])
                if stats.get("v_selected_absmax_after") is not None:
                    solve_v_absmax.setdefault(layer_idx, []).append(
                        stats["v_selected_absmax_after"]
                    )

            with torch.no_grad():
                if config.key_mode != "freeze":
                    k_param[0, head_idx].copy_(keys[n_frozen:].to(k_param.dtype))
                v_param[0, head_idx].copy_(
                    new_values[n_frozen:].to(v_param.dtype)
                )

            if mse is not None:
                layer_mses.append(mse)

        if layer_mses:
            mse_per_layer[layer_idx] = sum(layer_mses) / len(layer_mses)
            total_mse += mse_per_layer[layer_idx]

    mean_mse = total_mse / max(len(mse_per_layer), 1) if mse_per_layer else 0.0
    extra: dict = {}
    # Standing requirement (WORKERS.md): every AM run records the attention mass
    # landing on the written slots, per layer. Plus the query-supply ceiling that
    # `max_queries_per_head` is capped by (B-CASCADE).
    extra["max_queries_per_head"] = int(config.max_queries_per_head)
    if rope_theta != 10000.0:
        # Only recorded when moved off the historical default, so stock runs keep a
        # byte-identical `am_doc_*.pt` payload (B-ROPE).
        extra["rope_theta"] = rope_theta
    if n_queries_available:
        extra["n_queries_available_min"] = min(n_queries_available)
        extra["n_queries_available_max"] = max(n_queries_available)
        extra["n_queries_used_min"] = min(n_queries_used)
        extra["n_queries_used_max"] = max(n_queries_used)
    if solve_mass_on_S:
        extra["ref_mass_on_S_per_layer"] = {
            l: sum(v) / len(v) for l, v in solve_mass_on_S.items() if v
        }
    if solve_v_absmax:
        extra["v_selected_absmax_after_per_layer"] = {
            l: max(v) for l, v in solve_v_absmax.items() if v
        }
    if key_rewrite_info:
        # B-ROUTE key-side arm (MECH-005): how many of the `top_t` support slots per
        # (layer, head) actually received a DOCUMENT key, and how many were
        # counter-rotated. Only recorded when `key_mode != "freeze"`, so every
        # frozen-key run keeps a byte-identical `am_doc_*.pt` payload.
        per_layer_rw: dict[int, list[dict]] = {}
        for i in key_rewrite_info:
            per_layer_rw.setdefault(i["layer"], []).append(i)
        extra["key_rewrite"] = {
            "mode": config.key_mode,
            "reposition": key_reposition,
            "n_head_calls": len(key_rewrite_info),
            "n_selected_total": sum(i["n_selected"] for i in key_rewrite_info),
            "n_from_doc_total": sum(i["n_from_doc"] for i in key_rewrite_info),
            "n_from_cartridge_total": sum(
                i["n_from_cartridge"] for i in key_rewrite_info
            ),
            "n_changed_total": sum(i.get("n_changed", 0) for i in key_rewrite_info),
            "n_repositioned_total": sum(
                i.get("n_repositioned", 0) for i in key_rewrite_info
            ),
            "rope_delta": key_rewrite_info[0].get("rope_delta"),
        }
        extra["key_rewrite_per_layer"] = {
            l: {
                "n_heads": len(v),
                "n_selected": sum(i["n_selected"] for i in v),
                "n_from_doc": sum(i["n_from_doc"] for i in v),
                "n_changed": sum(i.get("n_changed", 0) for i in v),
                "n_repositioned": sum(i.get("n_repositioned", 0) for i in v),
            }
            for l, v in per_layer_rw.items()
        }
    if fit_beta and beta_fit_info:
        # B-SOLVE: the fitted beta distribution is the headline diagnostic of the
        # mass-matching arm. Only recorded when beta actually ran, so a stock run's
        # `am_doc_*.pt` payload stays byte-identical.
        extra["beta"] = {
            "box": beta_kwargs["beta_box"],
            "n_iters": beta_kwargs["n_iters"],
            "driver": beta_kwargs["nnls_driver"],
            "target_mode": beta_kwargs["target_mode"],
            "fit_scope": config.beta_fit_scope,
            "n_fits": len(beta_fit_info),
        }
        for key in (
            "rank_deficient",
            "warm_start_nonfinite",
            "warm_start_raised",
        ):
            extra["beta"][f"n_{key}"] = sum(
                1 for i in beta_fit_info if i.get(key)
            )
        ranks = [i["lstsq_rank"] for i in beta_fit_info if "lstsq_rank" in i]
        if ranks:
            extra["beta"]["lstsq_rank_min"] = min(ranks)
            extra["beta"]["lstsq_rank_max"] = max(ranks)
            extra["beta"]["n_cols"] = beta_fit_info[0].get("n_cols")
            extra["beta"]["n_rows"] = beta_fit_info[0].get("n_rows")
        for key in ("resid_clamp_frac", "resid_min", "frac_at_upper", "frac_at_lower"):
            vals = [i[key] for i in beta_fit_info if key in i]
            if vals:
                extra["beta"][f"{key}_mean"] = sum(vals) / len(vals)
                extra["beta"][f"{key}_max"] = max(vals)
        extra["beta_per_layer"] = {}
        all_beta = []
        for l, vs in beta_per_layer.items():
            b = torch.cat(vs)
            all_beta.append(b)
            extra["beta_per_layer"][l] = {
                "min": float(b.min().item()),
                "median": float(b.median().item()),
                "max": float(b.max().item()),
                "mean": float(b.mean().item()),
                "n": int(b.numel()),
            }
        if all_beta:
            b = torch.cat(all_beta)
            extra["beta"]["min"] = float(b.min().item())
            extra["beta"]["median"] = float(b.median().item())
            extra["beta"]["max"] = float(b.max().item())
            extra["beta"]["mean"] = float(b.mean().item())
    if oracle_write:
        extra["oracle_write"] = True
        extra["oracle_write_assign"] = getattr(config, "oracle_write_assign", "mass_ranked")
        extra["oracle_ref_mass_on_S_per_layer"] = {
            l: sum(v) / len(v) for l, v in oracle_mass_on_S.items() if v
        }
        extra["oracle_teacher_doc_mass_per_layer"] = {
            l: sum(v) / len(v) for l, v in oracle_doc_mass.items() if v
        }
        extra["oracle_n_written_min"] = min(oracle_n_written) if oracle_n_written else 0
        extra["oracle_n_written_max"] = max(oracle_n_written) if oracle_n_written else 0
    return AMUpdateStats(
        step=-1,
        mse_per_layer=mse_per_layer,
        mean_mse=mean_mse,
        n_queries=total_queries,
        mask=mask,
        extra=extra,
    )


def run_decoupled_tfidf_am_update(
    cache: nn.Module,
    wrapped_model: nn.Module,
    dataloader,
    tfidf_ranker: CacheTFIDFRanker,
    config: AttentionMatchingFinetuningConfig,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    local_rank: torch.device,
    step: int = 0,
    old_dataloader=None,
) -> tuple[AMUpdateStats, Optional[TFIDFRankingInfo]]:
    """Run a solve-only AM sparse update with one TF-IDF ranking pass.

    This is the corrected baseline for Phase 2 AM: gather reference queries once,
    rank sparse slots once, then apply one closed-form AM update. It deliberately
    avoids the optimizer-step loop used by gradient sparse finetuning.
    """
    was_training = wrapped_model.training
    wrapped_model.eval()

    t_collect = time.time()
    query_acc, target_acc, batch_count = _collect_reference_queries(
        wrapped_model,
        cache,
        dataloader,
        config,
        n_layers,
        n_kv_heads,
        head_dim,
        local_rank,
        collect_teacher_targets=config.target_mode == "teacher_attention",
    )
    old_query_acc = None
    old_batch_count = 0
    if old_dataloader is not None and config.old_reference_weight > 0:
        old_query_acc, _, old_batch_count = _collect_reference_queries(
            wrapped_model,
            cache,
            old_dataloader,
            config,
            n_layers,
            n_kv_heads,
            head_dim,
            local_rank,
        )

    collect_time = time.time() - t_collect

    t_rank = time.time()
    access_scores = query_acc.get_access_scores()
    old_access_scores = (
        old_query_acc.get_access_scores()
        if old_query_acc is not None
        else None
    )
    grad_mask, ranking_info = rank_am_slots(
        access_scores,
        config.top_t,
        tfidf_ranker,
        config,
        old_access_scores=old_access_scores,
        step=step,
    )
    rank_time = time.time() - t_rank

    t_solve = time.time()
    am_stats = apply_am_update_to_cache(
        cache,
        grad_mask,
        query_acc,
        n_layers=n_layers,
        head_dim=head_dim,
        target_mode=(
            config.target_mode
            if config.target_mode in ("self", "teacher_attention")
            else "self"
        ),
        ridge_lambda=config.ridge_lambda,
        freeze_keys=config.freeze_keys,
        max_queries_per_head=config.max_queries_per_head,
        compute_stats=config.compute_update_stats,
        old_query_accumulator=old_query_acc,
        target_accumulator=target_acc,
        old_reference_weight=config.old_reference_weight,
        delta_weight=config.delta_weight,
        ridge_scale=config.ridge_scale,
        ridge_lambda_min=config.ridge_lambda_min,
    )
    solve_time = time.time() - t_solve
    am_stats.step = step
    am_stats.timing = {
        "collect_s": collect_time,
        "rank_s": rank_time,
        "solve_s": solve_time,
        "total_s": collect_time + rank_time + solve_time,
        "n_ref_batches": batch_count,
        "n_old_ref_batches": old_batch_count,
    }

    if was_training:
        wrapped_model.train()

    logger.info(
        "Decoupled AM update complete: "
        f"batches={batch_count}, queries={am_stats.n_queries}, "
        f"mean_mse={am_stats.mean_mse:.6f}, timing={am_stats.timing}"
    )
    return am_stats, ranking_info
