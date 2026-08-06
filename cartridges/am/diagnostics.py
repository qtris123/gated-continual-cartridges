"""Stability probes for sparse AM Phase 2 writes.

Reusable diagnostics for:
- TF-IDF ranking vs absolute attention mass on selected support S
- design-matrix conditioning and effective ridge λ
- value-growth / reconstruction metrics across regularization variants

Read-only: nothing here is part of a write. Sweeps import it directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

from cartridges.am.components.objective import sparse_am_value_update
from cartridges.am.core import compute_attention_weights, effective_ridge_lambda


@dataclass
class SlotMassProbe:
    """Per-(layer, head) comparison of TF-IDF rank vs absolute attention mass."""

    layer_idx: int
    head_idx: int
    top_t: int
    mass_on_S_mean: float
    mass_on_S_min: float
    mass_on_S_p10: float
    abs_access_on_S_mean: float
    abs_access_global_mean: float
    abs_access_on_S_over_global: float
    tfidf_on_S_mean: float
    tfidf_global_mean: float
    tfidf_on_S_over_global: float
    # How often a high-TFIDF slot has low absolute access.
    frac_S_below_global_median_access: float
    frac_S_in_bottom_quartile_access: float
    spearman_tfidf_vs_abs_access: float


@dataclass
class SolveProbe:
    """Per-(layer, head) solve diagnostics for one regularization setting."""

    layer_idx: int
    head_idx: int
    ridge_scale: str
    ridge_lambda: float
    ridge_lambda_min: float
    delta_weight: float
    mass_on_S_mean: float
    x_smax: float
    x_smin: float
    x_cond: float
    spectral_lambda_raw: float
    effective_ridge_lambda: float
    mse_before: float
    mse_after: float
    residual_norm: float
    v_absmax_before: float
    v_absmax_after: float
    v_absmean_before: float
    v_absmean_after: float
    v_growth_max: float
    teacher_mass_on_doc_mean: float = 0.0


@dataclass
class DocProbeSummary:
    doc_index: int
    slug: str
    n_queries: int
    slot_mass: list[SlotMassProbe] = field(default_factory=list)
    solves: list[SolveProbe] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman correlation for 1-D tensors; returns nan if degenerate."""
    if a.numel() < 2:
        return float("nan")
    a = a.float().cpu()
    b = b.float().cpu()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / denom)


def probe_tfidf_vs_absolute_mass(
    *,
    layer_idx: int,
    head_idx: int,
    selected_indices: torch.Tensor,
    alpha: torch.Tensor,
    abs_access: torch.Tensor,
    tfidf: torch.Tensor,
    n_frozen: int = 0,
) -> SlotMassProbe:
    """Confirm whether TF-IDF top-t can have high relative score but tiny absolute mass.

    Args:
        selected_indices: selected trainable indices (before frozen offset) or full indices
            if already including frozen prefix.
        alpha: (n, T_full) attention weights over cartridge keys used in the AM solve
        abs_access: (T_trainable,) absolute access / TF numerator for this layer
        tfidf: (T_trainable,) TF-IDF scores for this layer
        n_frozen: if selected_indices are trainable-only, offset used for alpha columns
    """
    device = alpha.device
    selected = selected_indices.to(device=device, dtype=torch.long)
    # alpha may include frozen prefix; access/tfidf usually cover trainable only.
    if n_frozen > 0 and selected.min() < n_frozen:
        selected_alpha = selected
        selected_train = selected - n_frozen
        selected_train = selected_train[selected_train >= 0]
    elif n_frozen > 0:
        selected_alpha = selected + n_frozen
        selected_train = selected
    else:
        selected_alpha = selected
        selected_train = selected

    mass = alpha[:, selected_alpha].sum(dim=-1)
    abs_S = abs_access[selected_train].float()
    tfidf_S = tfidf[selected_train].float()
    abs_all = abs_access.float()
    tfidf_all = tfidf.float()
    global_median = abs_all.median()
    q25 = torch.quantile(abs_all, 0.25)

    return SlotMassProbe(
        layer_idx=layer_idx,
        head_idx=head_idx,
        top_t=int(selected_train.numel()),
        mass_on_S_mean=float(mass.mean().item()),
        mass_on_S_min=float(mass.min().item()),
        mass_on_S_p10=float(torch.quantile(mass, 0.10).item()),
        abs_access_on_S_mean=float(abs_S.mean().item()),
        abs_access_global_mean=float(abs_all.mean().item()),
        abs_access_on_S_over_global=float(
            (abs_S.mean() / abs_all.mean().clamp_min(1e-12)).item()
        ),
        tfidf_on_S_mean=float(tfidf_S.mean().item()),
        tfidf_global_mean=float(tfidf_all.mean().item()),
        tfidf_on_S_over_global=float(
            (tfidf_S.mean() / tfidf_all.mean().clamp_min(1e-12)).item()
        ),
        frac_S_below_global_median_access=float((abs_S < global_median).float().mean().item()),
        frac_S_in_bottom_quartile_access=float((abs_S <= q25).float().mean().item()),
        spearman_tfidf_vs_abs_access=_safe_spearman(tfidf_all, abs_all),
    )


def probe_sparse_solve(
    *,
    layer_idx: int,
    head_idx: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    targets: torch.Tensor,
    head_dim: int,
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
    ridge_lambda_min: float = 0.0,
    delta_weight: float = 0.0,
    attention_bias: Optional[torch.Tensor] = None,
    teacher_mass_on_doc_mean: float = 0.0,
) -> tuple[torch.Tensor, SolveProbe]:
    """Run one sparse AM solve and return updated values + probe metrics."""
    alpha = compute_attention_weights(
        queries, keys, head_dim, attention_bias=attention_bias
    )
    selected = selected_indices.to(device=keys.device, dtype=torch.long)
    X = alpha[:, selected].to(torch.float32)
    mass = X.sum(dim=-1)
    out_before = alpha @ values.to(torch.float32)
    mse_before = torch.nn.functional.mse_loss(out_before, targets.to(torch.float32)).item()

    svals = torch.linalg.svdvals(X)
    smax = float(svals[0].item()) if svals.numel() else 0.0
    smin = float(svals[-1].item()) if svals.numel() else 0.0
    cond = smax / max(smin, 1e-12)
    spectral_raw = effective_ridge_lambda(X, ridge_lambda=ridge_lambda, ridge_scale="spectral")

    new_values, stats = sparse_am_value_update(
        keys,
        values,
        queries,
        selected,
        targets=targets,
        head_dim=head_dim,
        ridge_lambda=ridge_lambda,
        ridge_scale=ridge_scale,
        ridge_lambda_min=ridge_lambda_min,
        delta_weight=delta_weight,
        compute_stats=True,
        attention_bias=attention_bias,
    )
    v_before = float(stats["v_selected_absmax_before"])
    v_after = float(stats["v_selected_absmax_after"])
    probe = SolveProbe(
        layer_idx=layer_idx,
        head_idx=head_idx,
        ridge_scale=ridge_scale,
        ridge_lambda=ridge_lambda,
        ridge_lambda_min=ridge_lambda_min,
        delta_weight=delta_weight,
        mass_on_S_mean=float(mass.mean().item()),
        x_smax=smax,
        x_smin=smin,
        x_cond=cond,
        spectral_lambda_raw=spectral_raw,
        effective_ridge_lambda=float(stats["effective_ridge_lambda"]),
        mse_before=mse_before,
        mse_after=float(stats["mse"]),
        residual_norm=float(stats["residual_norm"]),
        v_absmax_before=v_before,
        v_absmax_after=v_after,
        v_absmean_before=float(stats["v_selected_absmean_before"]),
        v_absmean_after=float(stats["v_selected_absmean_after"]),
        v_growth_max=v_after / max(v_before, 1e-8),
        teacher_mass_on_doc_mean=teacher_mass_on_doc_mean,
    )
    return new_values, probe


def summarize_slot_mass_probes(probes: list[SlotMassProbe]) -> dict[str, float]:
    if not probes:
        return {}
    mass = torch.tensor([p.mass_on_S_mean for p in probes])
    frac_med = torch.tensor([p.frac_S_below_global_median_access for p in probes])
    frac_q = torch.tensor([p.frac_S_in_bottom_quartile_access for p in probes])
    ratio_tfidf = torch.tensor([p.tfidf_on_S_over_global for p in probes])
    ratio_abs = torch.tensor([p.abs_access_on_S_over_global for p in probes])
    return {
        "n_heads": float(len(probes)),
        "mass_on_S_mean": float(mass.mean().item()),
        "mass_on_S_min_across_heads": float(mass.min().item()),
        "frac_heads_mass_lt_0_01": float((mass < 0.01).float().mean().item()),
        "frac_heads_mass_lt_0_05": float((mass < 0.05).float().mean().item()),
        "mean_frac_S_below_median_access": float(frac_med.mean().item()),
        "mean_frac_S_in_bottom_quartile_access": float(frac_q.mean().item()),
        "mean_tfidf_on_S_over_global": float(ratio_tfidf.mean().item()),
        "mean_abs_access_on_S_over_global": float(ratio_abs.mean().item()),
    }


def summarize_solve_probes(probes: list[SolveProbe]) -> dict[str, float]:
    if not probes:
        return {}
    growth = torch.tensor([p.v_growth_max for p in probes])
    vmax = torch.tensor([p.v_absmax_after for p in probes])
    mse = torch.tensor([p.mse_after for p in probes])
    lam = torch.tensor([p.effective_ridge_lambda for p in probes])
    mass = torch.tensor([p.mass_on_S_mean for p in probes])
    return {
        "n_heads": float(len(probes)),
        "mean_mass_on_S": float(mass.mean().item()),
        "mean_effective_lambda": float(lam.mean().item()),
        "min_effective_lambda": float(lam.min().item()),
        "mean_mse_after": float(mse.mean().item()),
        "max_mse_after": float(mse.max().item()),
        "mean_v_growth_max": float(growth.mean().item()),
        "max_v_growth_max": float(growth.max().item()),
        "mean_v_absmax_after": float(vmax.mean().item()),
        "max_v_absmax_after": float(vmax.max().item()),
        "frac_heads_v_growth_gt_10": float((growth > 10).float().mean().item()),
        "frac_heads_v_absmax_gt_100": float((vmax > 100).float().mean().item()),
    }
