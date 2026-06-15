"""
Unit tests for sparse cache finetuning momentum modes and key freezing.

Each test is self-contained: no model forward pass needed.  We build a
minimal TrainableCache, prime the SGD momentum buffers with one warm-up
step, then run a single masked step and assert the resulting param values.

Momentum mode semantics under test
-----------------------------------
soft    : non-top-t params drift because residual momentum is non-zero.
hard    : momentum zeroed before step → non-top-t params don't move.
freeze  : param + momentum both restored after step → exact no-change.
decouple: param restored after step → no drift, but momentum keeps decaying.

Key-freeze semantics
---------------------
freeze_keys=True  (default): key gradients zeroed → keys never update.
freeze_keys=False           : keys receive real gradients and update.
"""
from __future__ import annotations

import pytest
import torch
import torch.optim as optim

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.sparse_cache_finetuning import (
    apply_gradient_mask_to_cache,
    freeze_key_gradients,
    restore_non_top_t_state,
    save_non_top_t_state,
    zero_momentum_for_masked_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_LAYERS = 2
N_HEADS = 2
N_TOKENS = 16
HEAD_DIM = 8
TOP_T = 4
LR = 0.1
MOMENTUM = 0.9


def _make_cache(device: str = "cpu") -> TrainableCache:
    keys = [
        torch.randn(1, N_HEADS, N_TOKENS, HEAD_DIM)
        for _ in range(N_LAYERS)
    ]
    values = [
        torch.randn(1, N_HEADS, N_TOKENS, HEAD_DIM)
        for _ in range(N_LAYERS)
    ]
    cache = TrainableCache(
        config=AttnConfig(n_layers=N_LAYERS, n_heads=N_HEADS, head_dim=HEAD_DIM),
        init_keys=keys,
        init_values=values,
    )
    return cache.to(device)


def _prime_momentum(
    cache: TrainableCache,
    optimizer: optim.Optimizer,
    top_positions: torch.Tensor,
) -> None:
    """One warm-up step: all value positions get gradient=1, keys get 0."""
    for layer_idx in range(N_LAYERS):
        cache.trainable_values[layer_idx].grad = torch.ones(
            1, N_HEADS, N_TOKENS, HEAD_DIM
        )
        cache.trainable_keys[layer_idx].grad = torch.zeros(
            1, N_HEADS, N_TOKENS, HEAD_DIM
        )
    optimizer.step()
    optimizer.zero_grad()


def _set_masked_grads(
    cache: TrainableCache,
    top_positions: torch.Tensor,
) -> None:
    """
    Simulate the paper's gradient mask: top_positions get grad=1,
    everything else gets grad=0.  Keys always get grad=0 here; whether
    they are subsequently zeroed or not is controlled by freeze_key_gradients.
    """
    grad_mask = torch.zeros(1, N_HEADS, N_TOKENS, HEAD_DIM)
    grad_mask[:, :, top_positions, :] = 1.0
    for layer_idx in range(N_LAYERS):
        cache.trainable_values[layer_idx].grad = grad_mask.clone()
        cache.trainable_keys[layer_idx].grad = torch.ones(
            1, N_HEADS, N_TOKENS, HEAD_DIM
        )


def _non_top_slice(t: torch.Tensor, top_positions: torch.Tensor) -> torch.Tensor:
    all_pos = torch.arange(N_TOKENS)
    mask = torch.ones(N_TOKENS, dtype=torch.bool)
    mask[top_positions] = False
    return t[:, :, mask, :]


# ---------------------------------------------------------------------------
# Momentum masking tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("momentum_masking", ["soft", "hard", "freeze", "decouple"])
def test_momentum_masking_modes(momentum_masking: str):
    top_positions = torch.arange(TOP_T)  # positions 0..3 are "top"

    cache = _make_cache()
    optimizer = optim.SGD(cache.parameters(), lr=LR, momentum=MOMENTUM)

    # Prime so momentum buffers are non-zero before the real step.
    _prime_momentum(cache, optimizer, top_positions)

    # Snapshot param values to compare against after the step.
    v_before = [cache.trainable_values[l].data.clone() for l in range(N_LAYERS)]

    # Simulate the masked backward.
    _set_masked_grads(cache, top_positions)

    # Zero keys (they always have 0 effective gradient in the value-only setup).
    freeze_key_gradients(cache, N_LAYERS)

    # --- momentum handling ---
    pre_step_state = None
    if momentum_masking == "hard":
        zero_momentum_for_masked_positions(
            cache, top_positions, N_LAYERS, optimizer, freeze_keys=True
        )
    elif momentum_masking in ("freeze", "decouple"):
        pre_step_state = save_non_top_t_state(
            cache, top_positions, N_LAYERS, optimizer,
            save_momentum=(momentum_masking == "freeze"),
        )

    optimizer.step()

    if pre_step_state is not None:
        restore_non_top_t_state(
            cache, pre_step_state, N_LAYERS, optimizer,
            restore_momentum=(momentum_masking == "freeze"),
        )

    # --- assertions ---
    for layer_idx in range(N_LAYERS):
        v_after = cache.trainable_values[layer_idx].data

        top_changed = not torch.allclose(
            v_after[:, :, top_positions, :],
            v_before[layer_idx][:, :, top_positions, :],
        )
        assert top_changed, (
            f"[{momentum_masking}] layer {layer_idx}: "
            "top-t positions should have changed"
        )

        nt_after = _non_top_slice(v_after, top_positions)
        nt_before = _non_top_slice(v_before[layer_idx], top_positions)

        if momentum_masking == "soft":
            # Residual momentum means non-top-t params drift.
            assert not torch.allclose(nt_after, nt_before), (
                f"[soft] layer {layer_idx}: "
                "non-top-t positions should drift due to residual momentum"
            )

        elif momentum_masking == "hard":
            # Momentum zeroed → zero update → no drift at all.
            assert torch.allclose(nt_after, nt_before), (
                f"[hard] layer {layer_idx}: "
                "non-top-t positions must not change (momentum was zeroed)"
            )

        elif momentum_masking in ("freeze", "decouple"):
            # Param was explicitly restored → exact equality.
            assert torch.allclose(nt_after, nt_before), (
                f"[{momentum_masking}] layer {layer_idx}: "
                "non-top-t positions must be exactly restored"
            )


def test_freeze_vs_decouple_momentum_differ():
    """After the step, freeze keeps optimizer state intact; decouple lets it decay."""
    top_positions = torch.arange(TOP_T)

    def _run(mode: str):
        cache = _make_cache()
        optimizer = optim.SGD(cache.parameters(), lr=LR, momentum=MOMENTUM)
        _prime_momentum(cache, optimizer, top_positions)
        _set_masked_grads(cache, top_positions)
        freeze_key_gradients(cache, N_LAYERS)

        pre = save_non_top_t_state(
            cache, top_positions, N_LAYERS, optimizer,
            save_momentum=(mode == "freeze"),
        )
        optimizer.step()
        restore_non_top_t_state(
            cache, pre, N_LAYERS, optimizer,
            restore_momentum=(mode == "freeze"),
        )

        # Return the momentum buffer for the first value param's non-top positions.
        v_param = cache.trainable_values[0]
        buf = optimizer.state[v_param]["momentum_buffer"]
        return _non_top_slice(buf, top_positions).clone()

    freeze_buf = _run("freeze")
    decouple_buf = _run("decouple")

    # freeze: momentum was restored → unchanged from pre-step value.
    # decouple: momentum decayed by one step (×0.9) since grad=0.
    # They must be different.
    assert not torch.allclose(freeze_buf, decouple_buf), (
        "freeze and decouple should produce different post-step momentum buffers"
    )


# ---------------------------------------------------------------------------
# Key-freeze tests
# ---------------------------------------------------------------------------

def _run_key_step(freeze_keys: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """One SGD step; returns (keys_before, keys_after) for layer 0."""
    top_positions = torch.arange(TOP_T)

    cache = _make_cache()
    optimizer = optim.SGD(cache.parameters(), lr=LR, momentum=MOMENTUM)

    # Set non-zero gradients on both keys and values.
    _set_masked_grads(cache, top_positions)

    k_before = cache.trainable_keys[0].data.clone()

    if freeze_keys:
        freeze_key_gradients(cache, N_LAYERS)

    optimizer.step()

    return k_before, cache.trainable_keys[0].data.clone()


def test_freeze_keys_true_keys_unchanged():
    k_before, k_after = _run_key_step(freeze_keys=True)
    assert torch.allclose(k_before, k_after), (
        "freeze_keys=True: key params must not change"
    )


def test_freeze_keys_false_keys_change():
    k_before, k_after = _run_key_step(freeze_keys=False)
    assert not torch.allclose(k_before, k_after), (
        "freeze_keys=False: key params should update"
    )


# ---------------------------------------------------------------------------
# hard mode × freeze_keys interaction
# ---------------------------------------------------------------------------

def test_hard_freeze_keys_true_zeros_key_momentum():
    """hard + freeze_keys=True: key momentum must be zeroed."""
    top_positions = torch.arange(TOP_T)
    cache = _make_cache()
    optimizer = optim.SGD(cache.parameters(), lr=LR, momentum=MOMENTUM)
    _prime_momentum(cache, optimizer, top_positions)

    k_param = cache.trainable_keys[0]
    buf_before = optimizer.state[k_param]["momentum_buffer"].clone()

    _set_masked_grads(cache, top_positions)
    zero_momentum_for_masked_positions(
        cache, top_positions, N_LAYERS, optimizer, freeze_keys=True
    )

    buf_after = optimizer.state[k_param]["momentum_buffer"]
    assert torch.all(buf_after == 0), (
        "hard + freeze_keys=True: key momentum buffer must be all zeros"
    )


def test_hard_freeze_keys_false_preserves_key_momentum():
    """hard + freeze_keys=False: key momentum must be left untouched."""
    top_positions = torch.arange(TOP_T)
    cache = _make_cache()
    optimizer = optim.SGD(cache.parameters(), lr=LR, momentum=MOMENTUM)
    _prime_momentum(cache, optimizer, top_positions)

    k_param = cache.trainable_keys[0]
    buf_before = optimizer.state[k_param]["momentum_buffer"].clone()

    _set_masked_grads(cache, top_positions)
    zero_momentum_for_masked_positions(
        cache, top_positions, N_LAYERS, optimizer, freeze_keys=False
    )

    buf_after = optimizer.state[k_param]["momentum_buffer"]
    assert torch.allclose(buf_before, buf_after), (
        "hard + freeze_keys=False: key momentum must not be touched"
    )


# ---------------------------------------------------------------------------
# apply_gradient_mask_to_cache
# ---------------------------------------------------------------------------

def test_apply_gradient_mask_zeros_non_top_value_grads():
    """apply_gradient_mask_to_cache zeroes non-top-t value grads, keeps top-t."""
    top_positions = torch.arange(TOP_T)
    cache = _make_cache()

    # Set uniform gradients on all value positions.
    for layer_idx in range(N_LAYERS):
        cache.trainable_values[layer_idx].grad = torch.ones(
            1, N_HEADS, N_TOKENS, HEAD_DIM
        )
        cache.trainable_keys[layer_idx].grad = torch.ones(
            1, N_HEADS, N_TOKENS, HEAD_DIM
        )

    apply_gradient_mask_to_cache(cache, top_positions, N_LAYERS)

    for layer_idx in range(N_LAYERS):
        v_grad = cache.trainable_values[layer_idx].grad
        # top positions keep their gradient
        assert torch.all(v_grad[:, :, top_positions, :] == 1.0), (
            f"layer {layer_idx}: top-t value grads must remain 1"
        )
        # non-top positions are zeroed
        non_top = torch.ones(N_TOKENS, dtype=torch.bool)
        non_top[top_positions] = False
        assert torch.all(v_grad[:, :, non_top, :] == 0.0), (
            f"layer {layer_idx}: non-top-t value grads must be zeroed"
        )
        # key gradients are untouched by this function
        k_grad = cache.trainable_keys[layer_idx].grad
        assert torch.all(k_grad == 1.0), (
            f"layer {layer_idx}: key grads must not be touched by apply_gradient_mask_to_cache"
        )
