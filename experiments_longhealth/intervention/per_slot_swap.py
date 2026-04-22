"""Per-slot V-swap tracer.

For each (layer, slot) of trainable_values, temporarily replace the base
cartridge's V at that slot with the donor cartridge's V at the same slot,
run a packed teacher-forced forward over a batch of questions, and record
the aggregate token NLL (= log-perplexity). K is never touched.

Produces a `[num_layers, n_slots]` tensor of post-swap log-perplexity,
plus base / donor baselines. The restoration score per slot is then:

    R(L, p) = (log_ppl_base − log_ppl_swap(L,p))
              / (log_ppl_base − log_ppl_donor)

where base = B (forgotten) and donor = A (target).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _aggregate_nll(model, cache, batches, device) -> float:
    """Run teacher-forced forward over pre-packed batches; return total_nll/total_tokens."""
    total_nll = 0.0
    total_tok = 0
    for packed_ids, packed_seq, packed_pos, spans in batches:
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=packed_ids,
                    seq_ids=packed_seq,
                    position_ids=packed_pos,
                    past_key_values=cache,
                    use_cache=True,
                    mode="generate",
                )
        logits = out.logits[0]
        cache.clear()
        for s, a_start, a_end in spans:
            pred_pos = torch.arange(s + a_start - 1, s + a_end - 1, device=device)
            tgt_pos = torch.arange(s + a_start, s + a_end, device=device)
            nll = F.cross_entropy(
                logits[pred_pos].float(),
                packed_ids[tgt_pos],
                reduction="sum",
            )
            total_nll += nll.item()
            total_tok += (a_end - a_start)
    return total_nll / max(total_tok, 1)


def _pack_batches(dataset, n: int, batch_size: int, device: str):
    """Pre-pack the eval into a list of (packed_ids, packed_seq, packed_pos, spans)."""
    batches = []
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        ids_list, seq_list, pos_list, spans = [], [], [], []
        offset = 0
        for i in range(batch_start, batch_end):
            el = dataset[i]
            full_ids = torch.tensor(
                el.metadata["full_input_ids"], dtype=torch.long, device=device
            )
            spans.append((offset, el.metadata["answer_start"], el.metadata["answer_end"]))
            ids_list.append(full_ids)
            seq_list.append(torch.full_like(full_ids, i - batch_start))
            pos_list.append(torch.arange(len(full_ids), device=device))
            offset += len(full_ids)
        batches.append((
            torch.cat(ids_list),
            torch.cat(seq_list),
            torch.cat(pos_list),
            spans,
        ))
    return batches


def _pack_batches_with_qidx(dataset, n, batch_size, device):
    batches = []
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        ids_list, seq_list, pos_list, spans, q_idx = [], [], [], [], []
        offset = 0
        for i in range(batch_start, batch_end):
            el = dataset[i]
            full_ids = torch.tensor(
                el.metadata["full_input_ids"], dtype=torch.long, device=device
            )
            spans.append((offset, el.metadata["answer_start"], el.metadata["answer_end"]))
            q_idx.append(i)
            ids_list.append(full_ids)
            seq_list.append(torch.full_like(full_ids, i - batch_start))
            pos_list.append(torch.arange(len(full_ids), device=device))
            offset += len(full_ids)
        batches.append((
            torch.cat(ids_list), torch.cat(seq_list), torch.cat(pos_list), spans, q_idx,
        ))
    return batches


def _per_q_nll(model, cache, batches, n_q, device):
    nll = torch.zeros(n_q)
    tok = torch.zeros(n_q)
    for packed_ids, packed_seq, packed_pos, spans, q_idx in batches:
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=packed_ids,
                    seq_ids=packed_seq,
                    position_ids=packed_pos,
                    past_key_values=cache,
                    use_cache=True,
                    mode="generate",
                )
        logits = out.logits[0]
        cache.clear()
        for (s, a_start, a_end), i in zip(spans, q_idx):
            pred_pos = torch.arange(s + a_start - 1, s + a_end - 1, device=device)
            tgt_pos = torch.arange(s + a_start, s + a_end, device=device)
            l = F.cross_entropy(
                logits[pred_pos].float(), packed_ids[tgt_pos], reduction="sum"
            ).item()
            nll[i] += l
            tok[i] += (a_end - a_start)
    return nll / tok.clamp(min=1)


def per_layer_swap_trace_per_q(
    model,
    base_cache,
    donor_cache,
    dataset,
    device: str,
    batch_size: int = 32,
    log_every: int = 5,
):
    """Per-question per-layer A→B swap.

    For each layer L, replace base's entire V[L] with donor's V[L] (all slots,
    all heads), measure per-question log-perplexity, restore. Returns
    (base_per_q, donor_per_q, grid[n_q, num_layers]).
    """
    num_layers = len(base_cache.trainable_values)
    n = len(dataset)
    batches = _pack_batches_with_qidx(dataset, n, batch_size, device)

    base_per_q = _per_q_nll(model, base_cache, batches, n, device)
    donor_per_q = _per_q_nll(model, donor_cache, batches, n, device)
    print(f"  baseline base (B) mean = {base_per_q.mean():.4f}")
    print(f"  baseline donor (A) mean = {donor_per_q.mean():.4f}")

    grid = torch.zeros(n, num_layers)
    for L in range(num_layers):
        saved = base_cache.trainable_values[L].data.clone()
        base_cache.trainable_values[L].data.copy_(donor_cache.trainable_values[L].data)
        grid[:, L] = _per_q_nll(model, base_cache, batches, n, device)
        base_cache.trainable_values[L].data.copy_(saved)
        if (L + 1) % log_every == 0:
            print(f"  L={L + 1}/{num_layers}  mean_log_ppl={grid[:, L].mean():.4f}")
    return base_per_q, donor_per_q, grid


def per_token_slot_swap_trace_per_q(
    model,
    base_cache,
    donor_cache,
    dataset,
    device: str,
    batch_size: int = 32,
    log_every: int = 50,
):
    """Per-token-slot V-swap with per-question NLL tracking.

    Returns (base_per_q, donor_per_q, grid) where grid[q, p] is the log-perplexity
    on question q after swapping slot p's V across all layers from donor into base.
    """
    num_layers = len(base_cache.trainable_values)
    n_slots = base_cache.trainable_values[0].shape[2]
    n = len(dataset)
    batches = _pack_batches_with_qidx(dataset, n, batch_size, device)
    print(f"  packed {len(batches)} batches for {n} questions")

    base_per_q = _per_q_nll(model, base_cache, batches, n, device)
    donor_per_q = _per_q_nll(model, donor_cache, batches, n, device)
    print(f"  baseline base (B) mean log_ppl = {base_per_q.mean():.4f}")
    print(f"  baseline donor (A) mean log_ppl = {donor_per_q.mean():.4f}")

    grid = torch.zeros(n, n_slots)
    for p in range(n_slots):
        saved = [
            base_cache.trainable_values[L].data[:, :, p, :].clone()
            for L in range(num_layers)
        ]
        for L in range(num_layers):
            base_cache.trainable_values[L].data[:, :, p, :] = (
                donor_cache.trainable_values[L].data[:, :, p, :]
            )
        grid[:, p] = _per_q_nll(model, base_cache, batches, n, device)
        for L in range(num_layers):
            base_cache.trainable_values[L].data[:, :, p, :] = saved[L]
        if (p + 1) % log_every == 0:
            print(f"  p={p + 1}/{n_slots}  mean_log_ppl={grid[:, p].mean():.4f}")
    return base_per_q, donor_per_q, grid


def per_token_slot_swap_trace(
    model,
    base_cache,
    donor_cache,
    dataset,
    device: str,
    batch_size: int = 32,
    log_every: int = 50,
):
    """For each slot position p ∈ [0, n_slots), swap V at p from donor into base
    *across all layers simultaneously* (one 'virtual token' of the cartridge
    restored to phase-1). Measure aggregate nll on the dataset.

    Returns: (base_nll, donor_nll, nll_per_slot) where nll_per_slot has shape
    [n_slots].
    """
    num_layers = len(base_cache.trainable_values)
    n_slots = base_cache.trainable_values[0].shape[2]
    n = len(dataset)
    batches = _pack_batches(dataset, n, batch_size, device)
    print(f"  packed {len(batches)} batches for {n} questions")

    base_nll = _aggregate_nll(model, base_cache, batches, device)
    donor_nll = _aggregate_nll(model, donor_cache, batches, device)
    print(f"  baseline base (B) log_ppl = {base_nll:.4f}")
    print(f"  baseline donor (A) log_ppl = {donor_nll:.4f}")

    nlls = torch.zeros(n_slots)
    for p in range(n_slots):
        saved = [
            base_cache.trainable_values[L].data[:, :, p, :].clone()
            for L in range(num_layers)
        ]
        for L in range(num_layers):
            base_cache.trainable_values[L].data[:, :, p, :] = (
                donor_cache.trainable_values[L].data[:, :, p, :]
            )
        nll = _aggregate_nll(model, base_cache, batches, device)
        for L in range(num_layers):
            base_cache.trainable_values[L].data[:, :, p, :] = saved[L]
        nlls[p] = nll
        if (p + 1) % log_every == 0:
            print(f"  p={p + 1}/{n_slots}  log_ppl={nll:.4f}")
    return base_nll, donor_nll, nlls


def per_slot_swap_trace(
    model,
    base_cache,
    donor_cache,
    dataset,
    device: str,
    batch_size: int = 32,
    layers: Iterable[int] | None = None,
    log_every: int = 200,
):
    """Sweep (L, p) over trainable_values; for each, swap donor V at that slot into
    base, measure aggregate nll on the dataset, restore.

    Returns: (base_nll, donor_nll, ppl_grid) where ppl_grid has shape
    [len(layers), n_slots] containing post-swap aggregate nll per slot.
    """
    num_layers = len(base_cache.trainable_values)
    n_slots = base_cache.trainable_values[0].shape[2]
    layers = list(range(num_layers)) if layers is None else list(layers)

    n = len(dataset)
    batches = _pack_batches(dataset, n, batch_size, device)
    print(f"  packed {len(batches)} batches for {n} questions")

    base_nll = _aggregate_nll(model, base_cache, batches, device)
    donor_nll = _aggregate_nll(model, donor_cache, batches, device)
    print(f"  baseline base (B) log_ppl = {base_nll:.4f}")
    print(f"  baseline donor (A) log_ppl = {donor_nll:.4f}")

    grid = torch.zeros(len(layers), n_slots)
    step = 0
    total = len(layers) * n_slots
    for li, L in enumerate(layers):
        V_base = base_cache.trainable_values[L].data
        V_donor = donor_cache.trainable_values[L].data
        for p in range(n_slots):
            saved = V_base[:, :, p, :].clone()
            V_base[:, :, p, :] = V_donor[:, :, p, :]
            nll = _aggregate_nll(model, base_cache, batches, device)
            V_base[:, :, p, :] = saved
            grid[li, p] = nll
            step += 1
            if step % log_every == 0:
                print(f"  slot {step}/{total}  L={L} p={p}  log_ppl={nll:.4f}")
    return base_nll, donor_nll, grid
