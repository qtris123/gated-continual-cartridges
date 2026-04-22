"""Cartridge cache constructors for intervention experiments.

Each function returns a `TrainableCache` on the given device.

  load_cache          — plain baseline load from disk.
  build_swapped_cache — swap one or more layers of V from a donor cartridge
                        into the base cartridge (keys are not swapped).
  build_zeroed_cache  — zero out a single layer's V (and frozen_values, if any).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from cartridges.cache import TrainableCache


def load_cache(path: str, device: str) -> TrainableCache:
    return TrainableCache.from_pretrained(path, device=device).to(device)


def build_swapped_cache(
    base_path: str,
    donor_path: str,
    layer_idx,
    device: str,
) -> TrainableCache:
    """Swap one layer (int) or a set of layers (iterable) of V from donor into base.

    Keys are left untouched. Frozen values at the target layer(s) are also
    swapped when present.
    """
    layers: list[int]
    if isinstance(layer_idx, int):
        layers = [layer_idx]
    else:
        layers = list(layer_idx)

    merged = TrainableCache.from_pretrained(base_path, device=device)
    donor = TrainableCache.from_pretrained(donor_path, device=device)
    for L in layers:
        merged.trainable_values[L] = nn.Parameter(
            donor.trainable_values[L].data.clone(), requires_grad=False
        )
        if len(merged.frozen_values) > 0:
            merged.frozen_values[L] = nn.Parameter(
                donor.frozen_values[L].data.clone(), requires_grad=False
            )
    merged = merged.to(device)
    del donor
    torch.cuda.empty_cache()
    return merged


def build_zeroed_cache(
    cache_path: str,
    layer_idx: int | None,
    device: str,
) -> TrainableCache:
    """Load a cartridge. If `layer_idx` is given, zero out its V (trainable
    and frozen). If None, return the unmodified baseline cache."""
    cache = TrainableCache.from_pretrained(cache_path, device=device)
    if layer_idx is not None:
        cache.trainable_values[layer_idx] = nn.Parameter(
            torch.zeros_like(cache.trainable_values[layer_idx].data),
            requires_grad=False,
        )
        if len(cache.frozen_values) > 0:
            cache.frozen_values[layer_idx] = nn.Parameter(
                torch.zeros_like(cache.frozen_values[layer_idx].data),
                requires_grad=False,
            )
    return cache.to(device)
