"""Key selection (highest-attention / OMP), NNLS beta fitting, key rewrite."""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from cartridges.am.core import _attention_scores, _inv_sqrt_d


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


def select_keys_highest_attention(
    keys: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    head_dim: int,
    score_method: Literal["rms", "mean", "max"] = "rms",
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Select top-t keys by attention score (highest-attention-keys AM).

    Returns:
        C1: (t, d) selected keys
        beta: (t,) log-weights from NNLS on mass
        indices: list of selected key indices
    """
    scores = _attention_scores(
        queries,
        keys,
        head_dim,
        doc_key_start=doc_key_start,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )
    attn = F.softmax(scores, dim=-1)  # (n, T)

    if score_method == "rms":
        key_scores = torch.sqrt((attn ** 2).mean(dim=0))
    elif score_method == "mean":
        key_scores = attn.mean(dim=0)
    else:
        key_scores = attn.max(dim=0).values

    T = keys.shape[0]
    t = min(t, T)
    _, top_idx = torch.topk(key_scores, k=t, largest=True)
    indices = top_idx.tolist()

    C1 = keys[indices]

    # NNLS for beta (mass matching)
    exp_scores = torch.exp(scores - scores.max(dim=1, keepdim=True).values)
    target_mass = exp_scores.sum(dim=1)
    Phi = exp_scores[:, indices]
    w = nnls_projected_gradient(Phi, target_mass, n_iters=200)
    beta = torch.log(w.clamp(min=1e-12))

    return C1, beta, indices


def select_keys_omp(
    keys: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    head_dim: int,
    n_iters: int = 200,
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Greedy OMP key selection on attention-mass features (AM paper Algorithm 1).

    Args:
        keys: (T, d) candidate keys
        queries: (n, d) reference queries
        t: number of keys to select

    Returns:
        selected_keys: (t, d)
        beta: (t,) log-weights from NNLS mass fit
        indices: selected row indices into keys
    """
    scores = _attention_scores(
        queries,
        keys,
        head_dim,
        doc_key_start=doc_key_start,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )
    phi = torch.exp(scores - scores.max(dim=1, keepdim=True).values)  # (n, T)
    target_mass = phi.sum(dim=1)

    T = keys.shape[0]
    t = min(t, T)
    selected: list[int] = []
    residual = target_mass.clone()

    for _ in range(t):
        if not selected:
            col_scores = (phi.T @ target_mass).abs()
        else:
            phi_s = phi[:, selected]
            w = nnls_projected_gradient(phi_s, target_mass, n_iters=n_iters)
            residual = target_mass - phi_s @ w
            col_scores = (phi.T @ residual).abs()
            col_scores[selected] = -1.0

        next_idx = int(torch.argmax(col_scores).item())
        if next_idx in selected:
            remaining = [i for i in range(T) if i not in selected]
            if not remaining:
                break
            next_idx = remaining[0]
        selected.append(next_idx)

    phi_s = phi[:, selected]
    w = nnls_projected_gradient(phi_s, target_mass, n_iters=n_iters)
    beta = torch.log(w.clamp(min=1e-12))
    selected_keys = keys[selected]
    return selected_keys, beta, selected


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


def rewrite_keys_on_support(
    keys: torch.Tensor,
    selected_indices: torch.Tensor,
    candidate_keys: torch.Tensor,
    queries: torch.Tensor,
    mode: Literal["highest_attention", "omp"],
    head_dim: int,
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Replace cartridge key rows at selected_indices using teacher candidates.

    Args:
        keys: (T, d) current cartridge keys (single head)
        selected_indices: (t,) support indices in cartridge
        candidate_keys: (T_cand, d) pool, typically concat(cartridge, doc)
        queries: (n, d) reference queries

    Returns:
        new_keys: (T, d) with rows at selected_indices rewritten
    """
    device = keys.device
    dtype = keys.dtype
    selected_indices = selected_indices.to(device=device, dtype=torch.long)
    t = selected_indices.numel()
    if t == 0:
        return keys.clone()

    if doc_key_start is None and doc_rope_offset is not None:
        doc_key_start = candidate_keys.shape[0] - doc_rope_offset
    if mode == "highest_attention":
        new_k, _, sel_idx = select_keys_highest_attention(
            candidate_keys,
            queries,
            t,
            head_dim,
            score_method="rms",
            doc_key_start=doc_key_start,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
        )
    elif mode == "omp":
        new_k, _, sel_idx = select_keys_omp(
            candidate_keys,
            queries,
            t,
            head_dim,
            doc_key_start=doc_key_start,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
        )
    else:
        raise ValueError(f"Unsupported key rewrite mode: {mode}")

    out = keys.clone()
    out[selected_indices] = new_k.to(device=device, dtype=dtype)
    return out
