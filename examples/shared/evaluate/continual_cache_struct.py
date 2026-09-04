#!/usr/bin/env python3
"""Structural probe of the continual cartridge collapse.

Norms are flat across phases (see continual_cache_drift.py), so the generation
collapse is not a scale runaway. This tests the two structural hypotheses:

  (A) Repeated writes clobber a *shrinking, overlapping* slot set, so later
      phases overwrite earlier content instead of adding to it.
  (B) The written slots' keys (and/or values) collapse toward a *shared
      direction*, which makes cartridge attention query-independent -> a fixed
      low-entropy token gets read out -> the "the 2016 2016" / "100% of the"
      loops.

For each phase we mark the slots that differ from p01 ("written") and compare
written vs untouched slots on: key/value norm, and mean pairwise cosine
(direction concentration) per layer, plus cumulative coverage.

    python examples/shared/evaluate/continual_cache_struct.py --dataset finqa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_stack(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    k = torch.stack([t[0].float() for t in ckpt["trainable_keys"]])   # (L,H,S,D)
    v = torch.stack([t[0].float() for t in ckpt["trainable_values"]])
    return k, v


def mean_pairwise_cos(x):
    # x: (n, D) -> mean cosine over off-diagonal pairs
    if x.shape[0] < 2:
        return float("nan")
    xn = torch.nn.functional.normalize(x, dim=-1)
    g = xn @ xn.T
    n = g.shape[0]
    off = (g.sum() - g.diagonal().sum()) / (n * (n - 1))
    return off.item()


def layerwise_cos(stack, mask):
    # stack: (L,H,S,D), mask: (L,S) bool -> mean over layers/heads of pairwise cos
    L, H, S, D = stack.shape
    vals = []
    for l in range(L):
        idx = mask[l].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        for h in range(H):
            vals.append(mean_pairwise_cos(stack[l, h, idx]))
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def slot_norm_ls(stack):
    # (L,H,S,D) -> (L,S) norm averaged over heads
    return stack.norm(dim=-1).mean(dim=1)


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
        m = state / f"{p}.json"
        if m.exists():
            caches[p] = json.loads(m.read_text())["cache_path"]

    k0, v0 = load_stack(caches["p01"])
    L, H, S, D = v0.shape
    print(f"dataset={args.dataset}  L={L} H={H} S={S} D={D}\n")

    # baseline direction concentration on ALL p01 slots
    print("p01 baseline pairwise-cos over ALL slots: "
          f"key={layerwise_cos(k0, torch.ones(L,S,dtype=torch.bool)):.4f} "
          f"value={layerwise_cos(v0, torch.ones(L,S,dtype=torch.bool)):.4f}")

    cum_mask = torch.zeros(L, S, dtype=torch.bool)
    prev_k, prev_v = k0, v0
    header = f"\n{'phase':>5} {'cumcov%':>8} {'wKcos':>7} {'uKcos':>7} {'wVcos':>7} {'uVcos':>7} {'wVnorm':>7} {'uVnorm':>7} {'wKnorm':>7} {'uKnorm':>7}"
    print(header)
    for p in phases[1:]:
        if p not in caches:
            continue
        k, v = load_stack(caches[p])
        # slots changed vs p01 (cumulative write footprint)
        dv0 = (v - v0).norm(dim=-1).mean(dim=1)  # (L,S)
        cum_mask = dv0 > 1e-3
        unmask = ~cum_mask
        cov = cum_mask.float().mean().item() * 100
        vnorm = slot_norm_ls(v)
        knorm = slot_norm_ls(k)
        row = (
            f"{p:>5} {cov:8.1f} "
            f"{layerwise_cos(k, cum_mask):7.4f} {layerwise_cos(k, unmask):7.4f} "
            f"{layerwise_cos(v, cum_mask):7.4f} {layerwise_cos(v, unmask):7.4f} "
            f"{vnorm[cum_mask].mean():7.3f} {vnorm[unmask].mean():7.3f} "
            f"{knorm[cum_mask].mean():7.3f} {knorm[unmask].mean():7.3f}"
        )
        print(row)
        prev_k, prev_v = k, v

    print("\nlegend: w=written(vs p01) u=untouched; Kcos/Vcos=mean pairwise cosine "
          "(direction concentration); cumcov=cumulative % of (layer,slot) cells changed")


if __name__ == "__main__":
    main()
