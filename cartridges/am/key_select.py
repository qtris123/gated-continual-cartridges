"""Key selection (highest-attention / OMP), NNLS beta fitting, key rewrite."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from cartridges.am.core import _attention_scores, _inv_sqrt_d


def nnls_projected_gradient(
    Phi: torch.Tensor,
    target: torch.Tensor,
    n_iters: int = 100,
    lower_bound: float = 1e-12,
) -> torch.Tensor:
    """Solve min_{w>=0} ||Phi w - target||_2^2 via projected gradient descent.

    Args:
        Phi: (n, t)
        target: (n,)
        n_iters: PGD iterations
        lower_bound: minimum weight

    Returns:
        w: (t,) non-negative weights
    """
    n, t = Phi.shape
    device = Phi.device
    Phi = Phi.to(torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    try:
        w = torch.linalg.lstsq(Phi, target).solution.clamp_min(lower_bound)
    except RuntimeError:
        w = torch.ones(t, device=device, dtype=torch.float32)

    # Gradient Lipschitz constant for 1/2 ||Phi w - target||^2.
    L = torch.linalg.matrix_norm(Phi, ord=2).square().clamp_min(1e-8)
    lr = 1.0 / L

    for _ in range(n_iters):
        grad = Phi.T @ (Phi @ w - target)
        w = w - lr * grad
        w = torch.clamp(w, min=lower_bound)

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
) -> torch.Tensor:
    """Fit per-key log biases beta so student mass matches teacher log mass.

    Args:
        keys: (T, d)
        queries: (n, d)
        target_log_mass: (n,) teacher log unnormalized mass targets
        selected_indices: optional (t,) indices to fit
        base_beta: existing beta; preserved on non-selected positions

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
        residual_target = (target - fixed_mass).clamp_min(1e-12)
        phi = torch.exp(scores[:, idx] - row_shift[:, None])
        w = nnls_projected_gradient(phi, residual_target, n_iters=n_iters)
        beta = base_beta.clone()
        beta[idx] = torch.log(w.clamp(min=1e-12))
        return beta

    phi = torch.exp(scores - row_shift[:, None])
    w = nnls_projected_gradient(phi, target, n_iters=n_iters)
    return torch.log(w.clamp(min=1e-12))


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
