#!/usr/bin/env python3
"""Measure what accumulates in the cartridge tensors across continual phases.

Loads the p01..p05 caches of a lineage (paths read from the state markers) and,
per phase, reports value/key L2-norm distributions and, per stage transition,
how many slots moved and by how much. The question is whether the generation
collapse (fluent -> question-echo -> low-entropy loop) tracks a measurable
runaway in cartridge key/value scale on a shrinking set of repeatedly-rewritten
slots.

    python examples/shared/evaluate/continual_cache_drift.py --dataset finqa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_layers(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = [k.float() for k in ckpt["trainable_keys"]]
    vals = [v.float() for v in ckpt["trainable_values"]]
    return keys, vals


def slot_norms(tensors):
    # tensors: list over layers of (1, n_heads, n_slots, head_dim)
    # return per-layer (n_slots,) norm = L2 over head_dim, averaged over heads
    out = []
    for t in tensors:
        n = t[0].norm(dim=-1)  # (n_heads, n_slots)
        out.append(n.mean(dim=0))  # (n_slots,)
    return torch.stack(out)  # (n_layers, n_slots)


def summarize_norms(name, norms):
    flat = norms.flatten()
    print(
        f"  {name:>6}: mean={flat.mean():8.3f}  median={flat.median():8.3f}  "
        f"p99={flat.quantile(0.99):9.3f}  max={flat.max():10.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("finqa", "techqa", "qasper", "quality"))
    ap.add_argument("--tag", default="delta_ha_b0_idf0_ropefix_p01D")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    args = ap.parse_args()

    root = Path(args.root)
    state = root / f"outputs/{args.dataset}_5phase_state" / args.tag
    phases = ["p01", "p02", "p03", "p04", "p05"]
    caches = {}
    for p in phases:
        marker = state / f"{p}.json"
        if not marker.exists():
            continue
        caches[p] = json.loads(marker.read_text())["cache_path"]

    print(f"dataset={args.dataset}  tag={args.tag}")
    for p, path in caches.items():
        print(f"  {p}: {path}")

    prev_v = prev_k = None
    prev_p = None
    for p in phases:
        if p not in caches:
            continue
        keys, vals = load_layers(caches[p])
        vnorm = slot_norms(vals)
        knorm = slot_norms(keys)
        print(f"\n=== {p}  (n_layers={len(vals)}, n_slots={vals[0].shape[2]}) ===")
        summarize_norms("value", vnorm)
        summarize_norms("key", knorm)

        # stack raw slot vectors for delta: (n_layers, n_heads, n_slots, head_dim)
        v_stack = torch.stack([v[0] for v in vals])
        k_stack = torch.stack([k[0] for k in keys])
        if prev_v is not None:
            dv = (v_stack - prev_v).norm(dim=-1)  # (L, H, S)
            dk = (k_stack - prev_k).norm(dim=-1)
            # per (layer, slot) averaged over heads
            dv_ls = dv.mean(dim=1)  # (L, S)
            dk_ls = dk.mean(dim=1)
            moved = (dv_ls > 1e-4)
            n_slots = dv_ls.shape[1]
            frac_moved_per_layer = moved.float().mean(dim=1)  # (L,)
            # value magnitude on moved vs the pre-existing norm
            base_v = slot_norms(
                [pv.unsqueeze(0) for pv in prev_v]
            )  # reuse: prev per-slot norm
            print(
                f"  Δ{prev_p}->{p}: slots moved/layer mean={frac_moved_per_layer.mean()*n_slots:6.1f}"
                f"/{n_slots}  ({frac_moved_per_layer.mean()*100:4.1f}%)   "
                f"Δvalue mean={dv_ls[moved].mean():7.3f} max={dv_ls.max():8.3f}   "
                f"Δkey mean={dk_ls[dk_ls>1e-4].mean():7.3f} max={dk_ls.max():8.3f}"
            )
        prev_v, prev_k, prev_p = v_stack, k_stack, p


if __name__ == "__main__":
    main()
