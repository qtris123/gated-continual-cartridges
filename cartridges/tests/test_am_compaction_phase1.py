"""Fast CPU sanity for the classic-AM compaction building blocks (Phase 1).

Verifies per-(layer, head) teacher compaction produces finite values and a
reconstruction MSE well below an all-random baseline, without needing a model.
"""

import torch

from cartridges.am.compaction import compute_compaction_c2
from cartridges.am.core import compute_attention_output
from cartridges.am.key_select import select_keys_highest_attention

HEAD_DIM = 32
T_TEACHER = 256
NUM_TOKENS = 64
N_QUERIES = 48


def _compact_one_head(K_T, V_T, queries, enable_beta=True):
    C1, beta, idx = select_keys_highest_attention(
        K_T, queries, NUM_TOKENS, HEAD_DIM, score_method="rms"
    )
    if not enable_beta:
        beta = torch.zeros(C1.shape[0])
    C2 = compute_compaction_c2(
        C1, beta, K_T, V_T, queries, HEAD_DIM, ridge_lambda=1e-4, ridge_scale="spectral"
    )
    inv_sqrt_d = (1.0 / HEAD_DIM) ** 0.5
    sC = (queries @ C1.T).to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)
    approx = torch.softmax(sC, dim=-1) @ C2.to(torch.float32)
    target = compute_attention_output(queries, K_T, V_T, HEAD_DIM)
    return C1, C2, beta, approx, target, idx


def test_compaction_beats_random_baseline():
    torch.manual_seed(0)
    n_layers, n_kv_heads = 2, 3
    for _ in range(n_layers):
        for _ in range(n_kv_heads):
            K_T = torch.randn(T_TEACHER, HEAD_DIM)
            V_T = torch.randn(T_TEACHER, HEAD_DIM)
            queries = torch.randn(N_QUERIES, HEAD_DIM)

            C1, C2, beta, approx, target, idx = _compact_one_head(K_T, V_T, queries)

            assert torch.isfinite(C1).all()
            assert torch.isfinite(C2).all()
            assert torch.isfinite(beta).all()
            assert torch.isfinite(approx).all()
            assert C1.shape == (NUM_TOKENS, HEAD_DIM)
            assert C2.shape == (NUM_TOKENS, HEAD_DIM)
            assert len(set(idx)) == NUM_TOKENS  # distinct teacher keys

            mse_compact = torch.nn.functional.mse_loss(approx, target).item()
            # All-random baseline: predict values from the same distribution.
            random_out = torch.randn_like(target)
            mse_random = torch.nn.functional.mse_loss(random_out, target).item()
            # Predicting the per-dim mean baseline.
            mse_mean = torch.nn.functional.mse_loss(
                target.mean(dim=0, keepdim=True).expand_as(target), target
            ).item()

            assert mse_compact < mse_random
            assert mse_compact < mse_mean


def test_compaction_no_beta_finite():
    torch.manual_seed(1)
    K_T = torch.randn(T_TEACHER, HEAD_DIM)
    V_T = torch.randn(T_TEACHER, HEAD_DIM)
    queries = torch.randn(N_QUERIES, HEAD_DIM)
    C1, C2, beta, approx, target, idx = _compact_one_head(
        K_T, V_T, queries, enable_beta=False
    )
    assert torch.isfinite(C2).all()
    assert torch.allclose(beta, torch.zeros_like(beta))
    mse_compact = torch.nn.functional.mse_loss(approx, target).item()
    mse_mean = torch.nn.functional.mse_loss(
        target.mean(dim=0, keepdim=True).expand_as(target), target
    ).item()
    assert mse_compact < mse_mean


if __name__ == "__main__":
    test_compaction_beats_random_baseline()
    test_compaction_no_beta_finite()
    print("OK: compaction phase1 sanity passed")
