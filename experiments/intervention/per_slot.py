"""Per-slot V-ablation tracer.

For a single question, zero one (layer, slot) of the cartridge's V at a time
and measure the change in teacher-forced NLL on the gold answer span.

Produces a `[num_layers, n_slots]` tensor of Δnll. A sharp localization
hypothesis predicts a few hot (L, p) pairs carry most of the signal; a
diffuse picture means knowledge is superposed across many slots.

Scope: ablates `trainable_values` only (not frozen). K is left untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _single_q_nll(model, cache, element, device) -> float:
    full_ids = torch.tensor(
        element.metadata["full_input_ids"], dtype=torch.long, device=device
    )
    seq_ids = torch.zeros_like(full_ids)
    pos_ids = torch.arange(len(full_ids), device=device)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=full_ids,
                seq_ids=seq_ids,
                position_ids=pos_ids,
                past_key_values=cache,
                use_cache=True,
                mode="generate",
            )
    logits = out.logits[0]
    cache.clear()

    a_start = element.metadata["answer_start"]
    a_end = element.metadata["answer_end"]
    pred_pos = torch.arange(a_start - 1, a_end - 1, device=device)
    tgt_pos = torch.arange(a_start, a_end, device=device)
    tgt_ids = full_ids[tgt_pos]
    nll = F.cross_entropy(
        logits[pred_pos].float(), tgt_ids, reduction="mean"
    ).item()
    return nll


def per_slot_trace(
    model,
    cache,
    element,
    device: str,
    layers: list[int] | None = None,
    log_every: int = 500,
):
    """Sweep (L, p) over trainable_values; zero one slot at a time and record Δnll.

    Returns (baseline_nll: float, deltas: Tensor of shape [len(layers), n_slots]).
    """
    num_layers = len(cache.trainable_values)
    n_slots = cache.trainable_values[0].shape[2]
    layers = list(range(num_layers)) if layers is None else list(layers)

    baseline = _single_q_nll(model, cache, element, device)

    deltas = torch.zeros(len(layers), n_slots)
    step = 0
    total = len(layers) * n_slots
    for li, L in enumerate(layers):
        V = cache.trainable_values[L].data  # (1, n_heads, n_slots, head_dim)
        for p in range(n_slots):
            saved = V[:, :, p, :].clone()
            V[:, :, p, :] = 0
            nll = _single_q_nll(model, cache, element, device)
            V[:, :, p, :] = saved
            deltas[li, p] = nll - baseline
            step += 1
            if step % log_every == 0:
                print(f"  slot {step}/{total}  L={L} p={p}  Δnll={deltas[li, p].item():+.4f}")
    return baseline, deltas
