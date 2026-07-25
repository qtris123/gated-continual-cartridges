"""Value-only Attention Matching solves (frozen keys, residual lstsq)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from cartridges.am.core import (
    _inv_sqrt_d,
    _ridge_lstsq,
    compute_attention_output,
    compute_attention_weights,
    effective_ridge_lambda,
)


def sparse_am_value_update(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
    ridge_lambda_min: float = 0.0,
    delta_weight: float = 0.0,
    compute_stats: bool = True,
    attention_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Update only selected value rows to match attention targets.

    Uses residual formulation: fix non-selected values, solve for selected.

    Args:
        keys: (T, d) frozen keys
        values: (T, d) current values (only selected rows will change)
        queries: (n, d) reference queries
        selected_indices: (t,) long tensor of positions to update
        targets: (n, d) optional target outputs; if None, self-match pre-update
        head_dim: scaling dim (defaults to keys.shape[-1])
        ridge_lambda: ridge regularization for lstsq
        ridge_scale: how to scale ridge_lambda
        ridge_lambda_min: absolute floor on effective ridge coefficient
        delta_weight: trust-region weight on ||V_S - V_S_old||^2
        compute_stats: if False, avoid extra reconstruction work and host syncs

    Returns:
        new_values: (T, d) with only selected_indices rows updated
        stats: dict with reconstruction MSE, etc.
    """
    if head_dim is None:
        head_dim = keys.shape[-1]

    T, d = values.shape
    n = queries.shape[0]
    device = values.device
    dtype = values.dtype

    selected_indices = selected_indices.to(device=device, dtype=torch.long)
    if selected_indices.numel() == 0:
        return values.clone(), {"mse": 0.0, "n_queries": n, "n_selected": 0}

    # Full-cache attention weights (n, T)
    alpha = compute_attention_weights(queries, keys, head_dim, attention_bias=attention_bias)

    if targets is None:
        targets = alpha @ values.to(torch.float32)
    else:
        targets = targets.to(torch.float32)

    # Mask for selected / non-selected
    selected_mask = torch.zeros(T, dtype=torch.bool, device=device)
    selected_mask[selected_indices] = True

    # Fixed contribution from non-selected values
    alpha_ns = alpha[:, ~selected_mask]  # (n, T-t)
    V_ns = values[~selected_mask].to(torch.float32)  # (T-t, d)
    fixed = alpha_ns @ V_ns if alpha_ns.shape[1] > 0 else torch.zeros(n, d, device=device)

    residual = targets - fixed  # (n, d)

    # Design matrix: attention weights on selected positions only
    X = alpha[:, selected_mask]  # (n, t)
    t = X.shape[1]

    if t == 0:
        return values.clone(), {"mse": 0.0, "n_queries": n, "n_selected": 0}

    X_solve = X.to(torch.float32)
    Y_solve = residual
    if delta_weight > 0:
        scale = delta_weight ** 0.5
        X_solve = torch.cat(
            [X_solve, torch.eye(t, device=device, dtype=torch.float32) * scale],
            dim=0,
        )
        Y_solve = torch.cat(
            [Y_solve, values[selected_mask].to(torch.float32) * scale],
            dim=0,
        )

    # Solve X @ V_sel = residual for V_sel (t, d)
    V_sel_new = _ridge_lstsq(
        X_solve,
        Y_solve,
        ridge_lambda=ridge_lambda,
        ridge_scale=ridge_scale,
        ridge_lambda_min=ridge_lambda_min,
    )
    eff_lam = effective_ridge_lambda(
        X_solve,
        ridge_lambda=ridge_lambda,
        ridge_scale=ridge_scale,
        ridge_lambda_min=ridge_lambda_min,
    )

    new_values = values.clone()
    new_values[selected_mask] = V_sel_new.to(dtype)

    if not compute_stats:
        return new_values, {
            "mse": None,
            "n_queries": n,
            "n_selected": t,
            "residual_norm": None,
            "effective_ridge_lambda": eff_lam,
        }

    # Keys are frozen during sparse AM, so the attention weights are unchanged.
    output_new = compute_attention_weights(
        queries, keys, head_dim, attention_bias=attention_bias,
    ) @ new_values.to(torch.float32)
    mse = F.mse_loss(output_new, targets).item()

    v_old = values[selected_mask].to(torch.float32)
    v_new = new_values[selected_mask].to(torch.float32)
    stats = {
        "mse": mse,
        "n_queries": n,
        "n_selected": t,
        "residual_norm": residual.norm().item(),
        "effective_ridge_lambda": eff_lam,
        "mass_on_S_mean": X.sum(dim=-1).mean().item(),
        "v_selected_absmax_before": v_old.abs().max().item(),
        "v_selected_absmax_after": v_new.abs().max().item(),
        "v_selected_absmean_before": v_old.abs().mean().item(),
        "v_selected_absmean_after": v_new.abs().mean().item(),
        "v_delta_absmax": (v_new - v_old).abs().max().item(),
    }
    return new_values, stats


def guarded_sparse_am_value_update(
    keys: torch.Tensor,
    values: torch.Tensor,
    new_queries: torch.Tensor,
    selected_indices: torch.Tensor,
    old_queries: Optional[torch.Tensor] = None,
    targets_new: Optional[torch.Tensor] = None,
    targets_old: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
    ridge_lambda_min: float = 0.0,
    old_reference_weight: float = 1.0,
    delta_weight: float = 0.0,
    compute_stats: bool = True,
    attention_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Sparse AM update with old-reference preservation constraints.

    Solves a stacked residual least-squares system for selected value rows:

        ||A_new[:, S] V_S - R_new||^2
      + w_old ||A_old[:, S] V_S - R_old||^2
      + w_delta ||V_S - V_S_original||^2

    The old-reference term preserves old attention outputs when ``targets_old``
    is left as None. The new term can use an external target when available.
    """
    if head_dim is None:
        head_dim = keys.shape[-1]

    device = values.device
    dtype = values.dtype
    T, d = values.shape
    selected_indices = selected_indices.to(device=device, dtype=torch.long)
    if selected_indices.numel() == 0:
        return values.clone(), {
            "mse_new": 0.0 if compute_stats else None,
            "mse_old": 0.0 if compute_stats else None,
            "n_queries_new": new_queries.shape[0],
            "n_queries_old": 0 if old_queries is None else old_queries.shape[0],
            "n_selected": 0,
        }

    selected_mask = torch.zeros(T, dtype=torch.bool, device=device)
    selected_mask[selected_indices] = True

    def _design_and_residual(
        queries: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = compute_attention_weights(
            queries, keys, head_dim, attention_bias=attention_bias,
        )
        if targets is None:
            targets_f = alpha @ values.to(torch.float32)
        else:
            targets_f = targets.to(torch.float32)
        alpha_ns = alpha[:, ~selected_mask]
        v_ns = values[~selected_mask].to(torch.float32)
        fixed = (
            alpha_ns @ v_ns
            if alpha_ns.shape[1] > 0
            else torch.zeros(queries.shape[0], d, device=device)
        )
        return alpha[:, selected_mask], targets_f - fixed, targets_f

    new_queries = new_queries.to(device=device, dtype=keys.dtype)
    X_parts = []
    Y_parts = []

    X_new, R_new, target_new = _design_and_residual(new_queries, targets_new)
    X_parts.append(X_new)
    Y_parts.append(R_new)

    X_old = R_old = target_old = None
    if old_queries is not None and old_queries.numel() > 0 and old_reference_weight > 0:
        old_queries = old_queries.to(device=device, dtype=keys.dtype)
        X_old, R_old, target_old = _design_and_residual(old_queries, targets_old)
        scale = old_reference_weight ** 0.5
        X_parts.append(X_old * scale)
        Y_parts.append(R_old * scale)

    if delta_weight > 0:
        scale = delta_weight ** 0.5
        t = selected_mask.sum().item()
        X_parts.append(torch.eye(t, device=device, dtype=torch.float32) * scale)
        Y_parts.append(values[selected_mask].to(torch.float32) * scale)

    X = torch.cat(X_parts, dim=0)
    Y = torch.cat(Y_parts, dim=0)
    v_sel_new = _ridge_lstsq(
        X,
        Y,
        ridge_lambda=ridge_lambda,
        ridge_scale=ridge_scale,
        ridge_lambda_min=ridge_lambda_min,
    )

    new_values = values.clone()
    new_values[selected_mask] = v_sel_new.to(dtype)

    stats = {
        "mse_new": None,
        "mse_old": None,
        "n_queries_new": new_queries.shape[0],
        "n_queries_old": 0 if old_queries is None else old_queries.shape[0],
        "n_selected": selected_mask.sum().item(),
    }
    if compute_stats:
        output_new = compute_attention_weights(
            new_queries, keys, head_dim, attention_bias=attention_bias,
        ) @ new_values.to(torch.float32)
        stats["mse_new"] = F.mse_loss(output_new, target_new).item()
        if old_queries is not None and old_queries.numel() > 0 and target_old is not None:
            output_old = compute_attention_weights(
                old_queries, keys, head_dim, attention_bias=attention_bias,
            ) @ new_values.to(torch.float32)
            stats["mse_old"] = F.mse_loss(output_old, target_old).item()

    return new_values, stats


def refine_kv_head_values(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    head_dim: int,
    ridge_lambda: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    """Refine all value positions in-place (Phase 1, no compression)."""
    T = keys.shape[0]
    all_idx = torch.arange(T, device=keys.device)
    return sparse_am_value_update(
        keys, values, queries, all_idx,
        targets=None, head_dim=head_dim, ridge_lambda=ridge_lambda,
    )


def evaluate_attention_reconstruction(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    head_dim: int,
    reference_values: Optional[torch.Tensor] = None,
) -> dict:
    """Compute output MSE and cosine similarity between cache attention outputs."""
    ref_V = reference_values if reference_values is not None else values
    target = compute_attention_output(queries, keys, ref_V, head_dim)
    output = compute_attention_output(queries, keys, values, head_dim)

    mse = F.mse_loss(output, target).item()
    cos = F.cosine_similarity(output, target, dim=-1).mean().item()

    return {"output_mse": mse, "output_cosine": cos, "n_queries": queries.shape[0]}
