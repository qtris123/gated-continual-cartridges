"""DIAG-ROPE unit-level sanity for the opt-in AM_ROPE_THETA knob (CPU, no GPU).

Checks, in order:
  1. `_apply_rope_offset_to_queries` / `compute_teacher_targets` are unchanged at the
     historical default 10000.0 (bit-identical), and materially different at 5e6.
  2. `apply_document_am_write_to_cache` runs end-to-end on a tiny synthetic cache with
     both bases, returns finite MSE, and reproduces the default arm bit-identically
     when `rope_theta` is left at 10000.0.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from cartridges.am.core import _apply_rope_offset_to_queries
from cartridges.am.teacher import compute_teacher_targets
from cartridges.am.finetune import (
    AttentionMatchingFinetuningConfig,
    apply_document_am_write_to_cache,
)
from cartridges.am.query_accum import AMQueryAccumulator
from cartridges.sparse_cache_finetuning import GradientMask

import cartridges

print("CARTRIDGES_IMPORT_PATH=" + os.path.dirname(cartridges.__file__))

torch.manual_seed(0)
D = 128
N_LAYERS, N_KV, T_CART, T_DOC, NQ, TOP_T = 2, 2, 40, 4000, 24, 8

# ---------------------------------------------------------------- 1. primitives
q = torch.randn(NQ, D)
k = torch.randn(T_CART + T_DOC, D)
v = torch.randn(T_CART + T_DOC, D)

r_default = _apply_rope_offset_to_queries(q, T_DOC, D)
r_1e4 = _apply_rope_offset_to_queries(q, T_DOC, D, rope_theta=10000.0)
r_5e6 = _apply_rope_offset_to_queries(q, T_DOC, D, rope_theta=5000000.0)
assert torch.equal(r_default, r_1e4), "default rope_theta is no longer 10000.0"
cos = torch.nn.functional.cosine_similarity(r_1e4, r_5e6, dim=-1).mean().item()
print(f"[1] rotate offset={T_DOC}: default==1e4 bit-identical: True; "
      f"mean cos(1e4, 5e6) = {cos:.4f}")

t_default = compute_teacher_targets(q, k, v, D, n_cartridge_keys=T_CART,
                                    doc_rope_offset=T_DOC)
t_1e4 = compute_teacher_targets(q, k, v, D, n_cartridge_keys=T_CART,
                                doc_rope_offset=T_DOC, rope_theta=10000.0)
t_5e6 = compute_teacher_targets(q, k, v, D, n_cartridge_keys=T_CART,
                                doc_rope_offset=T_DOC, rope_theta=5000000.0)
assert torch.equal(t_default, t_1e4)
assert torch.isfinite(t_5e6).all()
rel = ((t_5e6 - t_1e4).norm() / t_1e4.norm()).item()
print(f"[1] teacher targets: default==1e4 bit-identical: True; "
      f"||T(5e6)-T(1e4)|| / ||T(1e4)|| = {rel:.4f}")


# --------------------------------------------------------- 2. end-to-end write
class TinyCache(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.trainable_keys = nn.ParameterList([
            nn.Parameter(torch.randn(1, N_KV, T_CART - 1, D, generator=g))
            for _ in range(N_LAYERS)
        ])
        self.trainable_values = nn.ParameterList([
            nn.Parameter(torch.randn(1, N_KV, T_CART - 1, D, generator=g))
            for _ in range(N_LAYERS)
        ])
        self.frozen_keys = nn.ParameterList([
            nn.Parameter(torch.randn(1, N_KV, 1, D, generator=g)) for _ in range(N_LAYERS)
        ])
        self.frozen_values = nn.ParameterList([
            nn.Parameter(torch.randn(1, N_KV, 1, D, generator=g)) for _ in range(N_LAYERS)
        ])
        self.trainable_beta = None
        self._num_frozen_tokens = 1


def run(rope_theta):
    torch.manual_seed(123)
    cache = TinyCache()
    mask = GradientMask(
        granularity="per_layer",
        positions_per_layer={l: torch.arange(TOP_T) for l in range(N_LAYERS)},
        n_tokens=T_CART - 1,
        top_t=TOP_T,
    )
    acc = AMQueryAccumulator(granularity="per_layer", n_layers=N_LAYERS, n_kv_heads=N_KV)
    for l in range(N_LAYERS):
        acc._queries[l].append(torch.randn(1, N_KV, NQ, D))
    doc_kv = {
        l: (torch.randn(N_KV, T_DOC, D), torch.randn(N_KV, T_DOC, D))
        for l in range(N_LAYERS)
    }
    kwargs = {} if rope_theta is None else {"rope_theta": rope_theta}
    cfg = AttentionMatchingFinetuningConfig(
        enabled=True, top_t=TOP_T, granularity="per_layer", key_mode="freeze",
        enable_beta=False, delta_weight=1e-2, ridge_lambda=1e-4, ridge_scale="spectral",
        max_queries_per_head=64, compute_update_stats=True, **kwargs,
    )
    stats = apply_document_am_write_to_cache(
        cache=cache, mask=mask, query_accumulator=acc, doc_kv=doc_kv,
        config=cfg, n_layers=N_LAYERS, head_dim=D,
    )
    vmax = max(float(v.detach().abs().max()) for v in cache.trainable_values)
    return stats, vmax


s_def, v_def = run(None)
s_1e4, v_1e4 = run(10000.0)
s_5e6, v_5e6 = run(5000000.0)

for name, s, vm in [("unset ", s_def, v_def), ("1e4   ", s_1e4, v_1e4),
                    ("5e6   ", s_5e6, v_5e6)]:
    mass = s.extra.get("ref_mass_on_S_per_layer", {})
    print(f"[2] {name} mean_mse={s.mean_mse:.10f} per_layer="
          f"{[round(x, 6) for x in s.mse_per_layer.values()]} "
          f"|v|max={vm:.4f} mass_on_S={[round(x, 5) for x in mass.values()]}")

assert s_def.mean_mse == s_1e4.mean_mse, "unset != explicit 10000.0 (not default-preserving)"
assert "rope_theta" not in s_def.extra and "rope_theta" not in s_1e4.extra
assert s_5e6.extra.get("rope_theta") == 5000000.0
assert all(map(lambda x: x == x, [s_def.mean_mse, s_5e6.mean_mse])), "NaN mse"
assert s_5e6.mean_mse != s_1e4.mean_mse, "rope_theta had no effect — knob is not wired"
print("[2] default-preserving: PASS (unset == explicit 10000.0, bit-identical mean_mse)")
print("[2] knob is live: PASS (5e6 changes mean_mse)")
print("SANITY_OK")
