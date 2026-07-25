"""Classic C1/C2 compaction building blocks (Phase 1 init)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from cartridges.am.core import (
    _inv_sqrt_d,
    _ridge_lstsq,
    compute_attention_output,
    compute_attention_weights,
)
from cartridges.am.key_select import select_keys_highest_attention


def naive_compaction_c2_update(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    head_dim: Optional[int] = None,
    ridge_lambda: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    """WRONG approach for sparse-in-full-cache: attention over subset only.

    Included for unit tests demonstrating why compaction C2 must not be used
    directly for sparse finetuning.
    """
    if head_dim is None:
        head_dim = keys.shape[-1]

    device = values.device
    dtype = values.dtype
    selected_indices = selected_indices.to(device=device, dtype=torch.long)

    C1 = keys[selected_indices]
    t = C1.shape[0]
    n = queries.shape[0]

    inv_sqrt_d = _inv_sqrt_d(head_dim)
    scores = (queries @ C1.T).to(torch.float32) * inv_sqrt_d
    X = F.softmax(scores, dim=-1)  # (n, t) — WRONG: ignores other keys

    targets = compute_attention_output(queries, keys, values, head_dim)

    V_sel_new = _ridge_lstsq(X, targets, ridge_lambda=ridge_lambda)

    new_values = values.clone()
    sel_mask = torch.zeros(keys.shape[0], dtype=torch.bool, device=device)
    sel_mask[selected_indices] = True
    new_values[sel_mask] = V_sel_new.to(dtype)

    alpha_new = compute_attention_weights(queries, keys, head_dim)
    output_new = alpha_new @ new_values.to(torch.float32)
    mse = F.mse_loss(output_new, targets).item()

    return new_values, {"mse": mse, "n_selected": t}


def compute_compaction_c2(
    C1: torch.Tensor,
    beta: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    head_dim: int,
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
) -> torch.Tensor:
    """Standard compaction C2 solve (valid when compacted block replaces full block).

    softmax(qK^T)V ≈ softmax(qC1^T + beta) C2
    """
    inv_sqrt_d = _inv_sqrt_d(head_dim)

    sK = (queries @ keys.T).to(torch.float32) * inv_sqrt_d
    attn_K = F.softmax(sK, dim=-1)
    Y = attn_K @ values.to(torch.float32)

    sC = (queries @ C1.T).to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)
    X = F.softmax(sC, dim=-1)

    return _ridge_lstsq(X, Y, ridge_lambda=ridge_lambda, ridge_scale=ridge_scale).to(values.dtype)


def compact_kv_head(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    head_dim: int,
    ridge_lambda: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, list, dict]:
    """Full AM compaction for one KV head (Phase 1 init).

    When t == T, selects all keys and refines values via C2 lstsq.

    Returns:
        new_keys: (t, d) — selected keys (subset of original)
        new_values: (t, d) — fitted values
        indices: selected indices in original cache
        stats: reconstruction metrics
    """
    T = keys.shape[0]
    t = min(t, T)

    if t == T:
        indices = list(range(T))
        C1 = keys
        beta = torch.zeros(T, device=keys.device, dtype=torch.float32)
    else:
        C1, beta, indices = select_keys_highest_attention(
            keys, queries, t, head_dim, score_method="rms"
        )

    C2 = compute_compaction_c2(
        C1, beta, keys, values, queries, head_dim, ridge_lambda=ridge_lambda
    )

    # Evaluate reconstruction on queries
    inv_sqrt_d = _inv_sqrt_d(head_dim)
    target = compute_attention_output(queries, keys, values, head_dim)

    sC = (queries @ C1.T).to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)
    X = F.softmax(sC, dim=-1)
    approx = X @ C2.to(torch.float32)
    mse = F.mse_loss(approx, target).item()

    stats = {"mse": mse, "t": t, "T": T, "n_queries": queries.shape[0]}
    return C1, C2, indices, stats
