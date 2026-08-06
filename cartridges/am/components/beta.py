"""Beta stage: per-key log-bias fitting so student attention mass matches the teacher.

Holds the projected-gradient NNLS solver plus the Phase-2 beta refit, split out
of ``key_select`` so that keys and beta are independent axes (B-SOLVE).
``components.keys`` imports ``nnls_projected_gradient`` from here: it is the
shared numeric primitive of both stages, and beta owns it because beta is the
only stage whose *output* is the NNLS solution.
"""

from __future__ import annotations

import math
from logging import getLogger
from typing import Literal, Optional

import torch
from pydrantic import ObjectConfig

from cartridges.am.core import _inv_sqrt_d

logger = getLogger(__name__)


def nnls_projected_gradient(
    Phi: torch.Tensor,
    target: torch.Tensor,
    n_iters: int = 100,
    lower_bound: float = 1e-12,
    upper_bound: Optional[float] = None,
    driver: Optional[str] = None,
    info: Optional[dict] = None,
) -> torch.Tensor:
    """Solve min_{lo<=w<=hi} ||Phi w - target||_2^2 via projected gradient descent.

    Args:
        Phi: (n, t)
        target: (n,)
        n_iters: PGD iterations
        lower_bound: minimum weight
        upper_bound: optional maximum weight (box constraint). ``None`` keeps the
            historical lower-bound-only projection.  B-SOLVE / LIT-002: the AM
            paper (App. C.2, Algorithm 3) *bounds both sides* -- ``w in [e^-3,
            e^3]`` for highest-attention keys, ``w <= e^7`` for OMP -- because an
            unbounded fit can drive a selected key to ``beta ~= -inf``, after
            which "the corresponding key cannot contribute to the attention
            output, regardless of Cv".
        driver: optional ``torch.linalg.lstsq`` driver for the warm start.  The
            torch default on CUDA is ``gels``, which *requires* full rank and
            returns undefined values (NaN/Inf) **without raising** otherwise --
            LIT-002's named root cause of the EXP-005b / EXP-006 NaNs, since the
            ``except RuntimeError`` below then never fires.  Pass ``'gelsd'`` for
            a rank-revealing SVD warm start.
        info: optional dict; filled in-place with warm-start diagnostics
            (``lstsq_rank``, ``rank_deficient``, ``warm_start_nonfinite``, ...).

    Returns:
        w: (t,) non-negative weights
    """
    n, t = Phi.shape
    device = Phi.device
    Phi = Phi.to(torch.float32)
    target = target.to(device=device, dtype=torch.float32)

    if info is not None:
        info["n_rows"] = int(n)
        info["n_cols"] = int(t)

    # Fail loudly on a poisoned input instead of laundering it into a NaN beta.
    if not (torch.isfinite(Phi).all() and torch.isfinite(target).all()):
        raise RuntimeError(
            "nnls_projected_gradient: non-finite input "
            f"(Phi finite={bool(torch.isfinite(Phi).all())}, "
            f"target finite={bool(torch.isfinite(target).all())}, "
            f"Phi shape={tuple(Phi.shape)})"
        )

    def _record_rank(sol) -> None:
        if info is None:
            return
        rank = getattr(sol, "rank", None)
        if rank is not None and rank.numel() > 0:
            info["lstsq_rank"] = int(rank.reshape(-1)[0].item())
            info["rank_deficient"] = bool(info["lstsq_rank"] < min(n, t))

    try:
        sol = (
            torch.linalg.lstsq(Phi, target)
            if driver is None
            else torch.linalg.lstsq(Phi, target, driver=driver)
        )
        w = sol.solution
        _record_rank(sol)
    except RuntimeError as exc:
        if driver is not None and Phi.is_cuda:
            # torch accepts ONLY driver='gels' for CUDA inputs -- the
            # rank-revealing drivers ('gelsd'/'gelss'/'gelsy') are CPU/LAPACK
            # only, and 'gels' is precisely the one that returns NaN without
            # raising on rank-deficient input (LIT-002). Phi here is tiny
            # (n x t, both <= a few hundred), so run the warm start on the CPU
            # and move the solution back.
            sol = torch.linalg.lstsq(Phi.cpu(), target.cpu(), driver=driver)
            w = sol.solution.to(device=device, dtype=torch.float32)
            _record_rank(sol)
            if info is not None:
                info["warm_start_on_cpu"] = True
        else:
            w = torch.ones(t, device=device, dtype=torch.float32)
            if info is not None:
                info["warm_start_raised"] = str(exc)[:200]

    if not torch.isfinite(w).all():
        # `gels` on a rank-deficient system returns NaN/Inf *silently*; a NaN here
        # survives every downstream clamp (torch.clamp passes NaN through), which
        # is exactly why EXP-006's output clamp could not work.  Fall back to the
        # paper's own uniform warm start and record it.
        if info is not None:
            info["warm_start_nonfinite"] = int((~torch.isfinite(w)).sum().item())
        w = torch.where(torch.isfinite(w), w, torch.ones_like(w))

    w = w.clamp_min(lower_bound)
    if upper_bound is not None:
        w = w.clamp_max(upper_bound)

    # Gradient Lipschitz constant for 1/2 ||Phi w - target||^2.
    L = torch.linalg.matrix_norm(Phi, ord=2).square().clamp_min(1e-8)
    lr = 1.0 / L

    for _ in range(n_iters):
        grad = Phi.T @ (Phi @ w - target)
        w = w - lr * grad
        w = torch.clamp(w, min=lower_bound, max=upper_bound)

    if not torch.isfinite(w).all():
        raise RuntimeError(
            "nnls_projected_gradient: non-finite solution after "
            f"{n_iters} PGD steps (Phi shape={tuple(Phi.shape)}, "
            f"box=[{lower_bound}, {upper_bound}], driver={driver})"
        )

    return w


def refit_beta_nnls(
    keys: torch.Tensor,
    queries: torch.Tensor,
    target_log_mass: torch.Tensor,
    head_dim: int,
    selected_indices: Optional[torch.Tensor] = None,
    base_beta: Optional[torch.Tensor] = None,
    n_iters: int = 200,
    beta_box: Optional[float] = None,
    nnls_driver: Optional[str] = None,
    target_mode: Literal["residual", "full"] = "residual",
    info: Optional[dict] = None,
) -> torch.Tensor:
    """Fit per-key log biases beta so student mass matches teacher log mass.

    Args:
        keys: (T, d)
        queries: (n, d)
        target_log_mass: (n,) teacher log unnormalized mass targets
        selected_indices: optional (t,) indices to fit
        base_beta: existing beta; preserved on non-selected positions
        n_iters: PGD iterations (AM paper App. C.2 uses 2 for highest-attention
            keys and 0 for OMP; this tree historically used 200)
        beta_box: optional symmetric box on the fitted log-weights, i.e.
            ``beta in [-beta_box, +beta_box]`` (paper: 3.0).  ``None`` keeps the
            historical unbounded-above / ``1e-12``-floored behaviour, whose floor
            puts beta at -27.6 -- the "effectively beta = -inf" state the paper
            writes a special rule to avoid (LIT-002).
        nnls_driver: lstsq driver for the NNLS warm start; ``'gelsd'`` is
            rank-revealing, the CUDA default ``gels`` returns NaN silently.
        target_mode: ``'residual'`` fits the selected keys against
            ``teacher_mass - mass(non-selected)``; ``'full'`` fits them against
            the full teacher mass, as the paper does over *all* compacted keys
            (LIT-002 divergence #3).
        info: optional dict; filled in-place with fit diagnostics.

    Returns:
        beta: (T,) float32 log-weights
    """
    T = keys.shape[0]
    device = keys.device
    inv_sqrt_d = _inv_sqrt_d(head_dim)
    scores = (queries @ keys.T).to(torch.float32) * inv_sqrt_d
    target_log_mass = target_log_mass.to(device=device, dtype=torch.float32)
    if base_beta is None:
        base_beta = torch.zeros(T, device=device, dtype=torch.float32)
    else:
        base_beta = base_beta.to(device=device, dtype=torch.float32)

    # Box on w: [e^-box, e^+box]. `None` reproduces the historical projection.
    if beta_box is None:
        w_lower, w_upper = 1e-12, None
    else:
        beta_box = float(beta_box)
        w_lower, w_upper = math.exp(-beta_box), math.exp(beta_box)
    if info is not None:
        info["beta_box"] = beta_box
        info["n_iters"] = int(n_iters)
        info["driver"] = nnls_driver
        info["target_mode"] = target_mode

    # Use one row-wise shift for both teacher target and student features.
    # Different shifts would destroy the absolute mass relation being fitted.
    max_student = (scores + base_beta).max(dim=1).values
    row_shift = torch.maximum(max_student, target_log_mass).detach()
    target = torch.exp(target_log_mass - row_shift)

    if selected_indices is not None and selected_indices.numel() > 0:
        idx = selected_indices.to(device=device, dtype=torch.long)
        selected_mask = torch.zeros(T, device=device, dtype=torch.bool)
        selected_mask[idx] = True
        fixed_mass = (
            torch.exp(
                scores[:, ~selected_mask]
                + base_beta[~selected_mask]
                - row_shift[:, None]
            ).sum(dim=1)
            if (~selected_mask).any()
            else torch.zeros_like(target)
        )
        raw_residual = target - fixed_mass
        residual_target = raw_residual.clamp_min(1e-12)
        if info is not None:
            # LIT-002 divergence #3: when the untouched slots over-supply mass the
            # clamp zeroes the WHOLE NNLS target -> w->0 -> beta->-27.6. Record how
            # often that actually happens rather than assuming it.
            info["resid_clamp_frac"] = float(
                (raw_residual <= 1e-12).float().mean().item()
            )
            info["resid_min"] = float(raw_residual.min().item())
        fit_target = target if target_mode == "full" else residual_target
        phi = torch.exp(scores[:, idx] - row_shift[:, None])
        w = nnls_projected_gradient(
            phi,
            fit_target,
            n_iters=n_iters,
            lower_bound=w_lower,
            upper_bound=w_upper,
            driver=nnls_driver,
            info=info,
        )
        beta = base_beta.clone()
        beta[idx] = torch.log(w.clamp(min=w_lower, max=w_upper))
        _record_beta_info(info, beta[idx], beta_box)
        return beta

    phi = torch.exp(scores - row_shift[:, None])
    w = nnls_projected_gradient(
        phi,
        target,
        n_iters=n_iters,
        lower_bound=w_lower,
        upper_bound=w_upper,
        driver=nnls_driver,
        info=info,
    )
    beta = torch.log(w.clamp(min=w_lower, max=w_upper))
    _record_beta_info(info, beta, beta_box)
    return beta


def _record_beta_info(
    info: Optional[dict],
    beta_fitted: torch.Tensor,
    beta_box: Optional[float],
) -> None:
    """Summarize a fitted beta vector into ``info`` (diagnostics only)."""
    if info is None:
        return
    b = beta_fitted.detach().float()
    info["beta_min"] = float(b.min().item())
    info["beta_median"] = float(b.median().item())
    info["beta_max"] = float(b.max().item())
    info["beta_mean"] = float(b.mean().item())
    info["beta_n"] = int(b.numel())
    if beta_box is not None:
        tol = 1e-6 * max(1.0, abs(beta_box))
        info["frac_at_upper"] = float((b >= beta_box - tol).float().mean().item())
        info["frac_at_lower"] = float((b <= -beta_box + tol).float().mean().item())


class BetaFitter:
    """Fits the per-key attention-mass bias beta (AM paper App. C.2)."""

    class Config(ObjectConfig):
        _pass_as_config = True

        # `None` = auto. Beta is an INDEPENDENT axis from `key_mode` (B-SOLVE):
        # auto means OFF, whatever the keys are doing. See `should_fit`.
        enabled: Optional[bool] = None
        fit_scope: Literal["all", "selected"] = "selected"
        # MECH-004 (AM paper App. C.2 / Algorithm 3). Defaults keep the pre-loop
        # behaviour; the paper's values are box=3.0, iters=2, driver='gelsd' (the
        # torch CUDA default `gels` returns NaN on rank-deficient input WITHOUT
        # raising). `target_mode='residual'` is this tree's own variant.
        # NB `target_mode`, not `target`: `ObjectConfig` already owns `target`.
        beta_box: Optional[float] = None
        nnls_iters: int = 200
        nnls_driver: Optional[str] = None
        target_mode: Literal["residual", "full"] = "residual"

    def __init__(self, config: Config):
        self.config = config

    def should_fit(self, *, key_mode: str) -> bool:
        """Whether to fit per-key log-bias beta. Decoupled from `key_mode` (B-SOLVE).

        Historically the unset fallback was `key_mode != "freeze"`, which
        (a) silently switched the beta/NNLS path ON for every key experiment and
        (b) made "beta with frozen keys" -- the configuration B-ROUTE actually needs
        -- expressible only by setting `ENABLE_BETA=1` explicitly. beta is now an
        independent axis: unset == off, whatever `key_mode` is. Every run in
        `state/results.csv` used `key_mode="freeze"`, where both rules agree, so no
        historical configuration changes behaviour.
        """
        if self.config.enabled is True:
            return True
        if self.config.enabled is False:
            return False
        if key_mode != "freeze":
            logger.warning(
                "ENABLE_BETA is unset with key_mode=%s: beta is OFF (it is now an "
                "independent axis; set ENABLE_BETA=1 to fit it).",
                key_mode,
            )
        return False

    def fit(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        target_log_mass: torch.Tensor,
        head_dim: int,
        *,
        selected_indices: Optional[torch.Tensor] = None,
        base_beta: Optional[torch.Tensor] = None,
        info: Optional[dict] = None,
    ) -> torch.Tensor:
        return refit_beta_nnls(
            keys,
            queries,
            target_log_mass,
            head_dim,
            selected_indices=selected_indices,
            base_beta=base_beta,
            n_iters=self.config.nnls_iters,
            beta_box=self.config.beta_box,
            nnls_driver=self.config.nnls_driver,
            target_mode=self.config.target_mode,
            info=info,
        )

    def describe(self) -> dict:
        """The fit settings recorded in the per-document stats payload."""
        return {
            "box": self.config.beta_box,
            "n_iters": self.config.nnls_iters,
            "driver": self.config.nnls_driver,
            "target_mode": self.config.target_mode,
            "fit_scope": self.config.fit_scope,
        }
