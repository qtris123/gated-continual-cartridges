#!/usr/bin/env python3
"""Per-stage slot-update geometry for a whole sweep stream, from the manifest.

For each arm of a stream it recovers the set of slots written at each stage and
reports cross-stage overlap (Jaccard) and coverage, then writes a JSON table and
a row-of-heatmaps figure. It prefers the native per-stage artifact emitted by the
write pipeline (`slots_written.pt`, produced for ANY selector); for legacy runs
that predate that artifact it falls back to diffing consecutive stage caches.

Because it is manifest-driven and keys off tags, a new selection function needs
NO change here: add the field to the component Config + a stream (manifest) and rerun.

CLI:
    python -m examples.shared.am.slot_geometry \\
        --manifest outputs/experiments/soft_locality/sweeps.yaml \\
        --stream lambda --dataset quality --which value \\
        --out-dir outputs/experiments/soft_locality/tables \\
        --fig outputs/experiments/soft_locality/slotgeom_lambda_quality.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from examples.shared.am.sweep import load_manifest, resolve
from examples.shared.paths import ROOT

N_STAGES = 5  # p01 base + writes p02..p05


def _state_dir(ds: str, tag: str) -> Path:
    return ROOT / f"outputs/{ds}_5phase_state" / tag


def _stage_cache_paths(ds: str, tag: str) -> list[Path] | None:
    """p01..p05 cache paths from the chain state markers (p01 = shared base)."""
    sdir = _state_dir(ds, tag)
    paths = []
    for k in range(1, N_STAGES + 1):
        j = sdir / f"p{k:02d}.json"
        if not j.exists():
            return None
        paths.append(Path(json.loads(j.read_text())["cache_path"]))
    return paths


def _native_masks(ds: str, tag: str) -> list[torch.Tensor] | None:
    """Per-stage boolean (n_layers, n_slots) masks from slots_written.pt sidecars.

    Stage k's sidecar sits in the run dir that produced its cache; the state
    marker p0k.json points at that cache. Returns masks for stages 2..5, or None
    if any sidecar is missing (caller falls back to cache-diff).
    """
    caches = _stage_cache_paths(ds, tag)
    if caches is None:
        return None
    masks = []
    for k in range(2, N_STAGES + 1):  # writes only
        sidecar = caches[k - 1].parent / "slots_written.pt"
        if not sidecar.exists():
            return None
        obj = torch.load(sidecar, map_location="cpu")
        stage = obj["stage"] if isinstance(obj, dict) else obj
        masks.append(stage > 0)
    return masks


def _diff_masks(ds: str, tag: str, which: str) -> list[torch.Tensor] | None:
    """Legacy fallback: slots changed between consecutive stage caches."""
    caches = _stage_cache_paths(ds, tag)
    if caches is None:
        return None
    key = {"value": "trainable_values", "key": "trainable_keys"}.get(which)
    loaded = [torch.load(p, map_location="cpu", weights_only=False) for p in caches]

    def changed(a, b):
        out = []
        for la, lb in zip(a, b):
            out.append((la != lb).any(dim=-1).any(dim=1).squeeze(0))
        return torch.stack(out)

    masks = []
    for k in range(1, len(loaded)):
        prev, cur = loaded[k - 1], loaded[k]
        if key is None:  # "either"
            mk = changed(prev["trainable_keys"], cur["trainable_keys"])
            mv = changed(prev["trainable_values"], cur["trainable_values"])
            masks.append(mk | mv)
        else:
            masks.append(changed(prev[key], cur[key]))
    return masks


def geometry(masks: list[torch.Tensor]) -> dict:
    ns = len(masks)
    cov = [m.float().mean().item() for m in masks]
    union = torch.zeros_like(masks[0])
    for m in masks:
        union |= m
    J = [[0.0] * ns for _ in range(ns)]
    for i in range(ns):
        for j in range(ns):
            inter = (masks[i] & masks[j]).float().sum(dim=1)
            uni = (masks[i] | masks[j]).float().sum(dim=1).clamp(min=1)
            J[i][j] = (inter / uni).mean().item()
    consec = sum(J[k][k + 1] for k in range(ns - 1)) / max(ns - 1, 1)
    return {
        "cov": cov,
        "union_cov": union.float().mean().item(),
        "jaccard": J,
        "consec": consec,
        "n_slots": int(masks[0].shape[1]),
        "n_layers": int(masks[0].shape[0]),
    }


def collect(manifest: dict, stream: str, dataset: str, which: str) -> dict:
    out = {}
    for arm in resolve(manifest, stream):
        masks = _native_masks(dataset, arm.tag)
        source = "native"
        if masks is None:
            masks = _diff_masks(dataset, arm.tag, which)
            source = "cache-diff"
        if masks is None:
            print(f"  [skip] {arm.label} ({arm.tag}): no state markers")
            continue
        g = geometry(masks)
        g.update(label=arm.label, tag=arm.tag, value=arm.value, source=source)
        out[arm.tag] = g
        cov = g["cov"]
        print(f"  {arm.label:<12} src={source:<10} union={g['union_cov']:.2f} "
              f"consecJ={g['consec']:.2f} cov={[round(c,2) for c in cov]}")
    return out


def plot(results: dict, arms, stream: str, dataset: str, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = [a for a in arms if a.tag in results]
    n = len(ordered)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    if n == 1:
        axes = [axes]
    labels = [f"s{i}" for i in range(2, N_STAGES + 1)]
    im = None
    for ax, arm in zip(axes, ordered):
        g = results[arm.tag]
        M = np.array(g["jaccard"])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.6 else "black")
        ax.set_title(f"{arm.label}\nunion={g['union_cov']:.2f} cJ={g['consec']:.2f}",
                     fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02,
                 label="cross-stage slot Jaccard (1=same, 0=disjoint)")
    plt.suptitle(f"{dataset}: per-stage update-slot overlap — stream '{stream}'",
                 fontweight="bold")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    print("wrote", fig_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--which", default="value", choices=["value", "key", "either"],
                    help="cache-diff fallback field; ignored when native artifact exists")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"===== slot geometry :: stream={args.stream} dataset={args.dataset} =====")
    results = collect(manifest, args.stream, args.dataset, args.which)

    if args.out_dir:
        out = Path(args.out_dir) / f"slotgeom_{args.stream}_{args.dataset}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print("wrote", out)
    if args.fig and results:
        plot(results, resolve(manifest, args.stream), args.stream, args.dataset,
             Path(args.fig))


if __name__ == "__main__":
    main()
