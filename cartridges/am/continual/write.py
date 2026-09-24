"""The per-document closed-form write: one pass over layers and KV heads.

For each (layer, head): build the ``[cartridge || doc]`` teacher, optionally
rewrite the keys on the selected support and refit beta, then solve for the new
value rows. Everything the write may vary is on ``AMStages``; this module owns
only the loop and the diagnostics it records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from cartridges.am.components.queries import AMQueryAccumulator
from cartridges.am.components.teacher import concat_teacher_kv
from cartridges.am.continual.config import AMStages
from cartridges.sparse_cache_finetuning import GradientMask

logger = getLogger(__name__)


@dataclass
class AMUpdateStats:
    step: int
    mse_per_layer: Dict[int, float]
    mean_mse: float
    n_queries: int
    mask: Optional[GradientMask] = None
    timing: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _onpolicy_refresh(
    refresh_fn: Callable[[int, int], tuple[AMQueryAccumulator, Optional[dict]]],
    layer_idx: int,
    group_size: int,
    old_accumulator: AMQueryAccumulator,
    doc_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_layers: int,
    events: list[dict],
) -> tuple[AMQueryAccumulator, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Re-extract reference queries (and optionally doc KV) from the updated cache.

    LIT-006 / SCOUT-AM divergence #3. Fail-loud: a refresh that returns the wrong
    shape, an empty layer, or non-finite queries would silently corrupt every
    remaining layer's solve, so all three are asserted here rather than surfacing
    as a mysterious MSE.

    Also records the **query drift** at the boundary layer -- mean cosine
    similarity and relative L2 change between the stale queries this layer would
    have been solved against and the on-policy ones -- which is the direct
    measurement of the activation shift this mechanism exists to remove.
    """
    t0 = time.time()
    result = refresh_fn(layer_idx, group_size)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise TypeError(
            "onpolicy_refresh_fn must return (query_accumulator, doc_kv_or_None), "
            f"got {type(result)!r}"
        )
    new_acc, new_doc_kv = result

    old_batches = old_accumulator._queries.get(layer_idx) or []
    new_batches = getattr(new_acc, "_queries", {}).get(layer_idx) or []
    if not new_batches:
        raise RuntimeError(
            f"on-policy refresh at layer {layer_idx} produced no reference "
            "queries; the write would fall back to nothing."
        )
    if len(new_batches) != len(old_batches):
        raise RuntimeError(
            f"on-policy refresh at layer {layer_idx} returned "
            f"{len(new_batches)} query batches, expected {len(old_batches)} "
            "(the reference dataloader must be replayed deterministically)."
        )
    q_new = new_batches[0]
    q_old = old_batches[0]
    if q_new.shape != q_old.shape:
        raise RuntimeError(
            f"on-policy refresh at layer {layer_idx} changed the query shape: "
            f"{tuple(q_old.shape)} -> {tuple(q_new.shape)}"
        )
    if not torch.isfinite(q_new).all():
        raise RuntimeError(
            f"on-policy refresh at layer {layer_idx} produced non-finite queries "
            f"({int((~torch.isfinite(q_new)).sum().item())} of {q_new.numel()})."
        )

    a = q_old.detach().float().reshape(-1, q_old.shape[-1])
    b = q_new.detach().float().reshape(-1, q_new.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel_l2 = (b - a).norm() / a.norm().clamp_min(1e-12)
    event = {
        "layer": int(layer_idx),
        "group_size": int(group_size),
        "n_batches": len(new_batches),
        "query_cos_mean": float(cos.mean().item()),
        "query_cos_min": float(cos.min().item()),
        "query_rel_l2": float(rel_l2.item()),
        "doc_kv_refreshed": new_doc_kv is not None,
    }

    if new_doc_kv is not None:
        for l in range(n_layers):
            if l not in new_doc_kv:
                raise RuntimeError(
                    f"on-policy doc-KV refresh is missing layer {l}"
                )
        k_new, v_new = new_doc_kv[layer_idx]
        k_old, v_old = doc_kv[layer_idx]
        if k_new.shape != k_old.shape or v_new.shape != v_old.shape:
            raise RuntimeError(
                f"on-policy doc-KV refresh changed shapes at layer {layer_idx}: "
                f"{tuple(k_old.shape)}/{tuple(v_old.shape)} -> "
                f"{tuple(k_new.shape)}/{tuple(v_new.shape)}"
            )
        if not (torch.isfinite(k_new).all() and torch.isfinite(v_new).all()):
            raise RuntimeError(
                f"on-policy doc-KV refresh produced non-finite KV at layer {layer_idx}"
            )
        event["doc_v_rel_l2"] = float(
            ((v_new.float() - v_old.float()).norm() / v_old.float().norm().clamp_min(1e-12)).item()
        )
        doc_kv = new_doc_kv

    event["refresh_s"] = time.time() - t0
    events.append(event)
    logger.info(
        "on-policy refresh before layer %d (group %d): cos=%.4f rel_l2=%.4f "
        "batches=%d doc_kv=%s %.2fs",
        layer_idx,
        group_size,
        event["query_cos_mean"],
        event["query_rel_l2"],
        event["n_batches"],
        event["doc_kv_refreshed"],
        event["refresh_s"],
    )
    return new_acc, doc_kv


def apply_document_am_write_to_cache(
    cache: nn.Module,
    mask: GradientMask,
    query_accumulator: AMQueryAccumulator,
    doc_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    stages: AMStages,
    n_layers: int,
    head_dim: int,
    old_query_accumulator: Optional[AMQueryAccumulator] = None,
    old_target_bank: Optional[dict[tuple[int, int], torch.Tensor]] = None,
    target_accumulator: Optional["AMTargetAccumulator"] = None,
    onpolicy_refresh_fn: Optional[
        Callable[[int, int], tuple[AMQueryAccumulator, Optional[dict]]]
    ] = None,
) -> AMUpdateStats:
    """Apply one per-document AM write with teacher [cartridge || doc KV].

    ``onpolicy_refresh_fn(layer_idx, group_size)`` is the LIT-006 hook: when
    ``stages.queries.config.onpolicy_layers > 0`` it is called at every group
    boundary and must return ``(fresh_query_accumulator, fresh_doc_kv_or_None)``
    re-extracted from the cache **as it stands after the previous groups were
    written**. It is only ever called when the flag is on, so a stock run is
    untouched.
    """
    device = cache.trainable_values[0].device
    mse_per_layer: dict[int, float] = {}
    total_mse = 0.0
    total_queries = 0

    objective_cfg = stages.objective.config
    max_queries_per_head = stages.queries.config.max_queries_per_head
    rope_theta = stages.rope_theta
    compute_stats = stages.compute_stats

    use_old_guard = (
        objective_cfg.enable_old_reference_guard
        and old_query_accumulator is not None
    )
    fit_beta = stages.beta.should_fit(key_mode=stages.keys.config.key_mode)
    beta_fit_info: list[dict] = []
    beta_per_layer: dict[int, list[torch.Tensor]] = {}
    key_rewrite_info: list[dict] = []
    oracle_write = objective_cfg.oracle_write
    oracle_mass_on_S: dict[int, list[float]] = {}
    oracle_doc_mass: dict[int, list[float]] = {}
    oracle_n_written: list[int] = []
    # B-CASCADE diagnostics (read-only; never touch the solution).
    solve_mass_on_S: dict[int, list[float]] = {}
    solve_v_absmax: dict[int, list[float]] = {}
    n_queries_available: list[int] = []
    n_queries_used: list[int] = []
    # LIT-006 on-policy layer-sequential re-extraction (off unless BOTH the config
    # field is > 0 and the caller supplied the hook).
    onpolicy_group = stages.queries.config.onpolicy_layers
    onpolicy_active = onpolicy_group > 0 and onpolicy_refresh_fn is not None
    if onpolicy_group > 0 and onpolicy_refresh_fn is None:
        raise ValueError(
            f"onpolicy_layers={onpolicy_group} but no `onpolicy_refresh_fn` was "
            "passed to apply_document_am_write_to_cache -- the on-policy write "
            "cannot silently fall back to stale queries."
        )
    onpolicy_events: list[dict] = []

    for layer_idx in range(n_layers):
        if onpolicy_active and layer_idx > 0 and layer_idx % onpolicy_group == 0:
            query_accumulator, doc_kv = _onpolicy_refresh(
                onpolicy_refresh_fn,
                layer_idx,
                onpolicy_group,
                query_accumulator,
                doc_kv,
                n_layers,
                events=onpolicy_events,
            )
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
            # The support is filled with this document's keys only. A short
            # note cannot occupy top_t slots; the highest-attention prefix is
            # rewritten and the rest of the cartridge stays as it was.
            n_doc_keys = int(k_doc.shape[0])
            if selected_full.numel() > n_doc_keys:
                selected_full = selected_full[:n_doc_keys]

            queries = query_accumulator.get_layer_head_queries(
                layer_idx, head_idx,
                n_q_heads=n_q_heads_actual,
                n_kv_heads=n_kv_heads,
            )
            flex_targets = None
            if target_accumulator is not None:
                flex_targets = target_accumulator.get_layer_head_targets(
                    layer_idx, head_idx,
                    n_q_heads=n_q_heads_actual,
                    n_kv_heads=n_kv_heads,
                )
                if flex_targets.numel() == 0:
                    flex_targets = None
            if queries.numel() == 0:
                continue
            queries = queries.to(device=device, dtype=keys.dtype)
            if flex_targets is not None:
                flex_targets = flex_targets.to(device=device, dtype=torch.float32)
                if flex_targets.shape[0] != queries.shape[0]:
                    raise RuntimeError(
                        f"Flex targets ({flex_targets.shape[0]}) are not aligned with "
                        f"queries ({queries.shape[0]}) at layer {layer_idx} head {head_idx}"
                    )

            # How many reference queries the accumulator could actually supply for
            # this (layer, head) BEFORE the subsample — this is the hard ceiling on
            # `max_queries_per_head` (B-CASCADE). Recorded, never padded.
            n_queries_available.append(int(queries.shape[0]))

            if queries.shape[0] > max_queries_per_head:
                idx = torch.randperm(queries.shape[0], device=device)[:max_queries_per_head]
                queries = queries[idx]
                if flex_targets is not None:
                    flex_targets = flex_targets[idx]

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
                    if old_queries.shape[0] > max_queries_per_head:
                        idx = torch.randperm(old_queries.shape[0], device=device)[
                            :max_queries_per_head
                        ]
                        old_queries = old_queries[idx]
                        if targets_old is not None:
                            targets_old = targets_old[idx]
                    # Preserve the pre-write cartridge behavior, including its
                    # current routing and beta, when keys or beta are rewritten.
                    if targets_old is None:
                        targets_old = stages.teacher.targets(
                            old_queries,
                            original_keys,
                            values,
                            head_dim,
                            attention_bias=base_beta,
                            rope_theta=rope_theta,
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
            if flex_targets is not None:
                # In-context flex outputs. The analytical qK^T target does not
                # match the residual Llama 3 uses once the document is visible.
                targets = flex_targets
            else:
                targets = stages.teacher.targets(
                    queries,
                    k_teacher,
                    v_teacher,
                    head_dim,
                    attention_bias=teacher_bias,
                    **teacher_rope_kwargs,
                )
            target_log_mass = stages.teacher.log_mass(
                queries,
                k_teacher,
                head_dim,
                attention_bias=teacher_bias,
                **teacher_rope_kwargs,
            )

            if stages.keys.enabled:
                # Select the new document's keys, as phase 1 does. Mixing the
                # old cartridge keys into the pool keeps the previous story in
                # the slots this question attends: rewriting every slot that
                # way still scored 4.1 nats, against 1.9 for a document-only
                # compaction of the same story. Slots outside this support
                # stay as they are.
                candidate_keys = k_doc
                rewrite_info: dict = {"layer": layer_idx, "head": head_idx}
                keys = stages.keys.write(
                    keys,
                    selected_full,
                    candidate_keys,
                    queries,
                    head_dim=head_dim,
                    doc_key_start=0,
                    doc_rope_offset=doc_rope_offset,
                    rope_theta=rope_theta,
                    # Clean prefill bakes document keys at m. Rebase onto the
                    # destination slot, the same move phase 1 uses.
                    doc_baked_pos_base=0,
                    info=rewrite_info,
                )
                key_rewrite_info.append(rewrite_info)

            head_beta = base_beta
            if fit_beta and beta_param is not None:
                fit_info: dict = {"layer": layer_idx, "head": head_idx}
                if stages.beta.config.fit_scope == "selected":
                    fit_idx = selected_full
                else:
                    fit_idx = torch.arange(
                        n_frozen,
                        keys.shape[0],
                        device=device,
                        dtype=torch.long,
                    )
                beta_full = stages.beta.fit(
                    keys,
                    queries,
                    target_log_mass,
                    head_dim,
                    selected_indices=fit_idx,
                    base_beta=base_beta,
                    info=fit_info,
                )
                if not torch.isfinite(beta_full).all():
                    # Fail loudly: a NaN beta would otherwise be written into the
                    # cache and poison the value solve (EXP-005b/EXP-006).
                    raise RuntimeError(
                        f"BetaFitter.fit returned non-finite beta at layer "
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

            new_values, stats = stages.objective.solve(
                keys, values, queries, selected_full,
                targets=targets,
                head_dim=head_dim,
                attention_bias=head_beta,
                old_queries=old_queries,
                targets_old=targets_old,
                has_old_reference=use_old_guard,
                v_doc=v_doc,
                k_teacher=k_teacher,
                teacher_bias=teacher_bias,
                n_cartridge_keys=n_cartridge_keys,
                doc_rope_offset=doc_rope_offset,
                rope_theta=rope_theta,
                compute_stats=compute_stats,
            )
            # The guarded rule reports `mse_new`; the plain and oracle rules `mse`.
            mse = stats["mse_new"] if "mse_new" in stats else stats["mse"]
            if oracle_write:
                oracle_mass_on_S.setdefault(layer_idx, []).append(stats["mass_on_S_mean"])
                if stats.get("teacher_doc_mass_mean") is not None:
                    oracle_doc_mass.setdefault(layer_idx, []).append(
                        stats["teacher_doc_mass_mean"]
                    )
                oracle_n_written.append(stats["n_written"])
            else:
                if stats.get("mass_on_S_mean") is not None:
                    solve_mass_on_S.setdefault(layer_idx, []).append(stats["mass_on_S_mean"])
                if stats.get("v_selected_absmax_after") is not None:
                    solve_v_absmax.setdefault(layer_idx, []).append(
                        stats["v_selected_absmax_after"]
                    )

            with torch.no_grad():
                if stages.keys.enabled:
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
    extra = _build_extra(
        stages=stages,
        n_queries_available=n_queries_available,
        n_queries_used=n_queries_used,
        solve_mass_on_S=solve_mass_on_S,
        solve_v_absmax=solve_v_absmax,
        onpolicy_events=onpolicy_events,
        onpolicy_group=onpolicy_group,
        key_rewrite_info=key_rewrite_info,
        fit_beta=fit_beta,
        beta_fit_info=beta_fit_info,
        beta_per_layer=beta_per_layer,
        oracle_write=oracle_write,
        oracle_mass_on_S=oracle_mass_on_S,
        oracle_doc_mass=oracle_doc_mass,
        oracle_n_written=oracle_n_written,
    )
    return AMUpdateStats(
        step=-1,
        mse_per_layer=mse_per_layer,
        mean_mse=mean_mse,
        n_queries=total_queries,
        mask=mask,
        extra=extra,
    )


def _build_extra(
    *,
    stages: AMStages,
    n_queries_available: list[int],
    n_queries_used: list[int],
    solve_mass_on_S: dict[int, list[float]],
    solve_v_absmax: dict[int, list[float]],
    onpolicy_events: list[dict],
    onpolicy_group: int,
    key_rewrite_info: list[dict],
    fit_beta: bool,
    beta_fit_info: list[dict],
    beta_per_layer: dict[int, list[torch.Tensor]],
    oracle_write: bool,
    oracle_mass_on_S: dict[int, list[float]],
    oracle_doc_mass: dict[int, list[float]],
    oracle_n_written: list[int],
) -> dict:
    """Assemble the ``am_doc_*.pt`` diagnostics payload.

    Every optional mechanism contributes its block ONLY when it actually ran, so
    a stock run's payload keeps exactly its historical keys.
    """
    extra: dict = {}
    # Standing requirement (WORKERS.md): every AM run records the attention mass
    # landing on the written slots, per layer. Plus the query-supply ceiling that
    # `max_queries_per_head` is capped by (B-CASCADE).
    extra["max_queries_per_head"] = int(stages.queries.config.max_queries_per_head)
    if stages.rope_theta != 10000.0:
        # Only recorded when moved off the historical default (B-ROPE).
        extra["rope_theta"] = stages.rope_theta
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
    if onpolicy_events:
        # LIT-006 / MECH-006.
        extra["onpolicy"] = {
            "group_size": onpolicy_group,
            "n_refreshes": len(onpolicy_events),
            "refresh_layers": [e["layer"] for e in onpolicy_events],
            "refresh_s_total": sum(e["refresh_s"] for e in onpolicy_events),
            "doc_kv_refreshed": bool(onpolicy_events[0]["doc_kv_refreshed"]),
            "query_cos_mean": sum(e["query_cos_mean"] for e in onpolicy_events)
            / len(onpolicy_events),
            "query_cos_min": min(e["query_cos_min"] for e in onpolicy_events),
            "query_rel_l2_mean": sum(e["query_rel_l2"] for e in onpolicy_events)
            / len(onpolicy_events),
            "query_rel_l2_max": max(e["query_rel_l2"] for e in onpolicy_events),
        }
        extra["onpolicy_events"] = onpolicy_events
    if key_rewrite_info:
        # B-ROUTE key-side arm (MECH-005): how many of the `top_t` support slots per
        # (layer, head) actually received a DOCUMENT key, and how many were
        # counter-rotated.
        per_layer_rw: dict[int, list[dict]] = {}
        for i in key_rewrite_info:
            per_layer_rw.setdefault(i["layer"], []).append(i)
        extra["key_rewrite"] = {
            "mode": stages.keys.config.key_mode,
            "reposition": stages.keys.config.key_reposition,
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
        # mass-matching arm.
        extra["beta"] = {**stages.beta.describe(), "n_fits": len(beta_fit_info)}
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
        extra["oracle_write_assign"] = stages.objective.config.oracle_write_assign
        extra["oracle_ref_mass_on_S_per_layer"] = {
            l: sum(v) / len(v) for l, v in oracle_mass_on_S.items() if v
        }
        extra["oracle_teacher_doc_mass_per_layer"] = {
            l: sum(v) / len(v) for l, v in oracle_doc_mass.items() if v
        }
        extra["oracle_n_written_min"] = min(oracle_n_written) if oracle_n_written else 0
        extra["oracle_n_written_max"] = max(oracle_n_written) if oracle_n_written else 0
    return extra
