"""Core attention math and ridge solver for Attention Matching.

Contains the low-level primitives shared by every AM solver:
- scaled attention scores / weights / outputs (with optional doc-RoPE offset)
- RoPE re-basing of post-RoPE keys and queries
- effective ridge lambda computation and the ridge least-squares solve

Nothing here knows about configs, caches, or stages: ``core`` is the only module
both ``initial/`` and ``continual/`` may depend on without going through
``components/``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _inv_sqrt_d(head_dim: int) -> float:
    return (1.0 / head_dim) ** 0.5


def _apply_rope_offset_to_queries(
    queries: torch.Tensor,
    offset: int,
    head_dim: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Rotate post-RoPE queries as if their absolute position increased by ``offset``."""
    if offset == 0:
        return queries
    device = queries.device
    dtype = queries.dtype
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = float(offset) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype=dtype).view(1, -1)
    sin = emb.sin().to(dtype=dtype).view(1, -1)
    x1 = queries[..., : head_dim // 2]
    x2 = queries[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return queries * cos + rotated * sin


def _rope_reposition(
    keys: torch.Tensor,
    from_pos: torch.Tensor,
    to_pos: torch.Tensor,
    head_dim: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Rotate each post-RoPE key row from absolute ``from_pos`` to ``to_pos``.

    Applies a per-row RoPE rotation by ``(to_pos - from_pos)`` so teacher keys,
    baked at their scattered document positions, are re-based onto sequential
    cartridge slot positions (KVFromText-consistent geometry) for eval.
    """
    device = keys.device
    delta = (to_pos.to(torch.float32) - from_pos.to(torch.float32))  # (t,)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angle = delta[:, None] * inv_freq[None, :]  # (t, d/2)
    emb = torch.cat([angle, angle], dim=-1)  # (t, d)
    cos = emb.cos().to(keys.dtype)
    sin = emb.sin().to(keys.dtype)
    x1 = keys[..., : head_dim // 2]
    x2 = keys[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return keys * cos + rotated * sin


def _attention_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Scaled qK^T scores, optionally with a document RoPE offset on the doc block."""
    inv_sqrt_d = _inv_sqrt_d(head_dim)
    if (
        doc_key_start is None
        or doc_rope_offset is None
        or doc_rope_offset == 0
        or doc_key_start >= keys.shape[0]
    ):
        return (queries @ keys.T).to(torch.float32) * inv_sqrt_d

    k_cartridge = keys[:doc_key_start]
    k_doc = keys[doc_key_start:]
    scores_c = (queries @ k_cartridge.T).to(torch.float32) * inv_sqrt_d
    q_doc = _apply_rope_offset_to_queries(
        queries,
        doc_rope_offset,
        head_dim,
        rope_theta=rope_theta,
    )
    scores_d = (q_doc @ k_doc.T).to(torch.float32) * inv_sqrt_d
    return torch.cat([scores_c, scores_d], dim=-1)


def compute_attention_weights(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
    attention_bias: Optional[torch.Tensor] = None,
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Softmax attention weights over all keys.

    Args:
        queries: (n, d)
        keys: (T, d)
        head_dim: scaling dimension
        attention_bias: optional (T,) or (n, T) additive bias before softmax
        doc_key_start: first index of document keys when teacher uses [C||D]
        doc_rope_offset: extra RoPE positions between cartridge queries and doc keys

    Returns:
        weights: (n, T) float32
    """
    scores = _attention_scores(
        queries,
        keys,
        head_dim,
        doc_key_start=doc_key_start,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )
    if attention_bias is not None:
        bias = torch.broadcast_to(
            attention_bias.to(torch.float32), scores.shape
        )
        scores = scores + bias
    return F.softmax(scores, dim=-1)


def compute_attention_output(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    head_dim: int,
    attention_bias: Optional[torch.Tensor] = None,
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Attention output softmax(qK^T)V for all queries."""
    weights = compute_attention_weights(
        queries,
        keys,
        head_dim,
        attention_bias,
        doc_key_start=doc_key_start,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )
    return weights @ values.to(torch.float32)


def effective_ridge_lambda(
    X: torch.Tensor,
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
    ridge_lambda_min: float = 0.0,
) -> float:
    """Compute the absolute ridge coefficient used by ``_ridge_lstsq``."""
    if ridge_lambda == 0:
        return 0.0
    n, k = X.shape
    if ridge_scale == "spectral":
        try:
            lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2) ** 2).item()
        except Exception:
            lam = ridge_lambda * (
                (torch.linalg.matrix_norm(X, ord="fro") ** 2) / max(k, 1)
            ).item()
    elif ridge_scale == "frobenius":
        lam = ridge_lambda * (
            (torch.linalg.matrix_norm(X, ord="fro") ** 2) / max(k, 1)
        ).item()
    elif ridge_scale in ("fixed", "absolute"):
        lam = ridge_lambda
    else:
        raise ValueError(f"Unknown ridge_scale: {ridge_scale}")
    if ridge_lambda_min > 0:
        lam = max(lam, ridge_lambda_min)
    return float(lam)


def _ridge_lstsq(
    X: torch.Tensor,
    Y: torch.Tensor,
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
    ridge_lambda_min: float = 0.0,
) -> torch.Tensor:
    """Solve min ||X @ W - Y||_F^2 + lam * ||W||_F^2.

    Args:
        X: (n, k) design matrix (attention weights on selected positions)
        Y: (n, d) targets
        ridge_lambda: base regularization
        ridge_scale: 'spectral', 'frobenius', or 'fixed'
        ridge_lambda_min: absolute floor on the effective ridge coefficient

    Returns:
        W: (k, d) solution
    """
    n, k = X.shape
    dtype_out = Y.dtype
    lam = effective_ridge_lambda(
        X,
        ridge_lambda=ridge_lambda,
        ridge_scale=ridge_scale,
        ridge_lambda_min=ridge_lambda_min,
    )

    try:
        if lam == 0:
            W = torch.linalg.lstsq(X, Y, driver="gels").solution
        elif n < k:
            XXt = X @ X.T
            XXt = 0.5 * (XXt + XXt.T)
            XXt.diagonal().add_(lam)
            L = torch.linalg.cholesky(XXt)
            Z = torch.cholesky_solve(Y, L)
            W = X.T @ Z
        else:
            XtX = X.T @ X
            XtX = 0.5 * (XtX + XtX.T)
            XtX.diagonal().add_(lam)
            L = torch.linalg.cholesky(XtX)
            XtY = X.T @ Y
            W = torch.cholesky_solve(XtY, L)
    except Exception:
        lam = max(lam, 1e-6)
        if n < k:
            XXt = X @ X.T
            XXt = 0.5 * (XXt + XXt.T)
            XXt.diagonal().add_(lam)
            L = torch.linalg.cholesky(XXt)
            Z = torch.cholesky_solve(Y, L)
            W = X.T @ Z
        else:
            XtX = X.T @ X
            XtX = 0.5 * (XtX + XtX.T)
            XtX.diagonal().add_(lam)
            L = torch.linalg.cholesky(XtX)
            XtY = X.T @ Y
            W = torch.cholesky_solve(XtY, L)

    if torch.isnan(W).any():
        raise RuntimeError("NaNs in ridge lstsq solution")

    return W.to(dtype_out)
