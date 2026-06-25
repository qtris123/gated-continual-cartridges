"""
Top-t slot selection investigation for sparse-cache cartridges (Llama, Qasper).

For each continual_sparse run, we:
  1. load `sparse_slot_log.pt` (one snapshot every ~30 optimizer steps),
  2. decode each entry into a boolean (sub_component, slot) selection mask,
  3. pick the snapshot closest to the median training step ("mid-training"),
  4. compute per-component median consecutive-step Jaccard as a stability stat,
  5. render:
      - main_snapshot_grid.png      (3 granularity rows x 4 sparsity cols, mid-training)
      - per_head_full_grid.png      (full 28x8 per_head heatmap per top_t)
      - appendix/<gran>_3batch.png  (three batches at ~25% / 50% / 75% of training)

All outputs land in `<repo>/qasper-top-t-slot-investigation/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
INVEST_DIR = REPO_ROOT / "qasper-top-t-slot-investigation"
FIG_DIR = INVEST_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(FIG_DIR / "appendix").mkdir(parents=True, exist_ok=True)


# Map (granularity, top_t) -> launch directory under outputs/.
# We resolve the run UUID by globbing for sparse_slot_log.pt below.
RUNS = {
    ("global", 32):    "2026-06-22-15-29-24-continual_sparse_global_top-32-key-value_all-reduce_num-tokens-512",
    ("global", 64):    "2026-06-22-15-29-40-continual_sparse_global_top-64-key-value_all-reduce_num-tokens-512",
    ("global", 128):   "2026-06-22-15-29-58-continual_sparse_global_top-128-key-value_all-reduce_num-tokens-512",
    ("global", 256):   "2026-06-22-15-30-11-continual_sparse_global_top-256-key-value_all-reduce_num-tokens-512",
    ("per_layer", 32): "2026-06-22-12-05-37-continual_sparse-per-layer-top-32-key-value_all-reduce_num-tokens-512",
    ("per_layer", 64): "2026-06-22-12-06-23-continual_sparse-per-layer-top-64-key-value_all-reduce_num-tokens-512",
    ("per_layer", 128):"2026-06-22-12-09-37-continual_sparse-per-layer-top-128-key-value_all-reduce_num-tokens-512",
    ("per_layer", 256):"2026-06-22-12-09-53-continual_sparse-per-layer-top-256-key-value_all-reduce_num-tokens-512",
    ("per_head", 32):  "2026-06-22-10-54-19-continual_sparse-per-head-top-32-key-value_all-reduce_num-tokens-512",
    ("per_head", 64):  "2026-06-22-10-54-36-continual_sparse-per-head-top-64-key-value_all-reduce_num-tokens-512",
    ("per_head", 128): "2026-06-22-10-54-50-continual_sparse-per-head-top-128-key-value_all-reduce_num-tokens-512",
    ("per_head", 256): "2026-06-22-10-55-05-continual_sparse-per-head-top-256-key-value_all-reduce_num-tokens-512",
}

GRANULARITIES = ["global", "per_layer", "per_head"]
TOP_TS = [32, 64, 128, 256]

# Reference layer used for the compact per_head panel in the main figure.
# Layer 14 ~= mid-stack for Llama-3.2-3B (28 layers) and is the same reference
# layer used in notebook/granularity_overlap_analysis.py.
REF_LAYER = 14


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def find_slot_log(run_dir_name: str) -> Optional[Path]:
    run_root = OUTPUTS / run_dir_name
    if not run_root.is_dir():
        return None
    candidates = sorted(run_root.glob("*/sparse_slot_log.pt"))
    return candidates[0] if candidates else None


def shapes_from_log(log: list) -> tuple[int, int, int]:
    """Infer (n_tokens, n_layers, n_kv_heads) from access_scores shape."""
    a = log[0]["access_scores"]
    gran = log[0]["granularity"]
    if gran == "global":
        return int(a.shape[-1]), 1, 1
    if gran == "per_layer":
        return int(a.shape[-1]), int(a.shape[0]), 1
    if gran == "per_head":
        return int(a.shape[-1]), int(a.shape[0]), int(a.shape[1])
    raise ValueError(gran)


def entry_to_mask(entry, n_tokens: int, n_layers: int, n_kv_heads: int) -> torch.Tensor:
    """Decode a `sparse_slot_log` entry into a boolean selection mask.

    Returned shapes match `granularity`:
      global    -> (n_tokens,)
      per_layer -> (n_layers, n_tokens)
      per_head  -> (n_layers, n_kv_heads, n_tokens)
    """
    gran = entry["granularity"]
    pos = entry["top_positions"]

    if gran == "global":
        mask = torch.zeros(n_tokens, dtype=torch.bool)
        mask[pos[0].long()] = True
        return mask

    if gran == "per_layer":
        mask = torch.zeros(n_layers, n_tokens, dtype=torch.bool)
        for l, p in pos.items():
            mask[l, p.long()] = True
        return mask

    if gran == "per_head":
        mask = torch.zeros(n_layers, n_kv_heads, n_tokens, dtype=torch.bool)
        for l, p in pos.items():
            mask[l].scatter_(-1, p.long(), True)
        return mask

    raise ValueError(gran)


def jaccard_last_dim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-component Jaccard along the last dimension; returns shape a.shape[:-1]."""
    inter = (a & b).sum(-1).float()
    union = (a | b).sum(-1).float()
    return torch.where(union > 0, inter / union, torch.ones_like(inter))


# ----------------------------------------------------------------------
# Load and summarise
# ----------------------------------------------------------------------
def load_all_runs() -> dict:
    run_data: dict = {}
    for (gran, top_t), launch_dir in RUNS.items():
        path = find_slot_log(launch_dir)
        if path is None:
            print(f"[WARN] missing sparse_slot_log.pt for ({gran}, t={top_t}) in {launch_dir}")
            continue

        log = torch.load(path, weights_only=False)
        n_tokens, n_layers, n_kv_heads = shapes_from_log(log)
        masks = [entry_to_mask(e, n_tokens, n_layers, n_kv_heads) for e in log]
        steps = [int(e["step"]) for e in log]

        run_data[(gran, top_t)] = {
            "path": path,
            "log": log,
            "masks": masks,
            "steps": steps,
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
        }
        print(
            f"[load] {gran:<9} t={top_t:<3}  "
            f"snapshots={len(log):>3}  steps={steps[0]}..{steps[-1]}  "
            f"shape=(L={n_layers}, H={n_kv_heads}, N={n_tokens})"
        )

    return run_data


def add_mid_step(run_data: dict):
    for rd in run_data.values():
        steps = rd["steps"]
        target = (max(steps) + min(steps)) / 2
        mid_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - target))
        rd["mid_idx"] = mid_idx
        rd["mid_step"] = steps[mid_idx]


def stability_stats(run_data: dict) -> dict:
    stats: dict = {}
    for k, rd in run_data.items():
        masks = rd["masks"]
        per_step = []
        for i in range(1, len(masks)):
            j = jaccard_last_dim(masks[i - 1], masks[i])
            per_step.append(float(j.float().mean().item()))
        arr = np.array(per_step) if per_step else np.array([np.nan])
        stats[k] = {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_consecutive_pairs": int(len(per_step)),
        }
        rd["stability"] = stats[k]
    return stats


def save_stability_table(run_data: dict, path: Path):
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow([
            "granularity", "top_t", "n_snapshots", "step_first", "step_last", "mid_step",
            "median_jaccard", "mean_jaccard", "min_jaccard", "max_jaccard",
            "n_consecutive_pairs",
        ])
        for k in sorted(run_data.keys(), key=lambda x: (GRANULARITIES.index(x[0]), x[1])):
            rd = run_data[k]
            s = rd["stability"]
            steps = rd["steps"]
            w.writerow([
                k[0], k[1], len(rd["masks"]), steps[0], steps[-1], rd["mid_step"],
                f"{s['median']:.3f}", f"{s['mean']:.3f}",
                f"{s['min']:.3f}", f"{s['max']:.3f}",
                s["n_consecutive_pairs"],
            ])
    print(f"[save] {path}")


def save_stability_markdown(run_data: dict, path: Path):
    lines = [
        "# Slot-selection stability (Llama, Qasper)",
        "",
        "`J̃` = median across consecutive logged snapshots of the per-component Jaccard ",
        "between top-t selections.  Higher = more stable (snapshot tells the whole story); ",
        "lower = snapshot is one of many.",
        "",
        "|  granularity  | top-t |  mid-step  |  J̃ median | J̃ mean | J̃ min | J̃ max | pairs |",
        "| ------------- | ----- | ---------- | --------- | ------ | ----- | ----- | ----- |",
    ]
    for k in sorted(run_data.keys(), key=lambda x: (GRANULARITIES.index(x[0]), x[1])):
        rd = run_data[k]
        s = rd["stability"]
        lines.append(
            f"| {k[0]:<13} | {k[1]:<5} | {rd['mid_step']:<10} | "
            f"{s['median']:.3f}     | {s['mean']:.3f}  | "
            f"{s['min']:.3f} | {s['max']:.3f} | {s['n_consecutive_pairs']:<5} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))
    print(f"[save] {path}")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def _mask_for_plot(rd: dict, snap_idx: int) -> np.ndarray:
    """Project an entry's mask onto a 2-D image (component, slot) for plotting.

    For per_head we pick `REF_LAYER` so the panel stays small.
    """
    mask = rd["masks"][snap_idx]
    if mask.dim() == 1:
        return mask.unsqueeze(0).float().numpy()  # (1, N)
    if mask.dim() == 2:
        return mask.float().numpy()
    return mask[REF_LAYER].float().numpy()


def plot_main_snapshot(run_data: dict, save_path: Path):
    fig, axes = plt.subplots(
        3, 4,
        figsize=(22, 10),
        gridspec_kw={"height_ratios": [0.4, 2.5, 1.0]},
    )

    for col_idx, t in enumerate(TOP_TS):
        # Row 0 — global
        rd = run_data.get(("global", t))
        ax = axes[0, col_idx]
        if rd is not None:
            img = _mask_for_plot(rd, rd["mid_idx"])
            ax.imshow(img, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(
                f"global · t={t} · step {rd['mid_step']} · "
                f"J\u0303={rd['stability']['median']:.2f}",
                fontsize=10,
            )
            ax.set_yticks([])
        else:
            ax.set_visible(False)
        if col_idx == 0:
            ax.set_ylabel("global", fontsize=11)

        # Row 1 — per_layer (28 rows)
        rd = run_data.get(("per_layer", t))
        ax = axes[1, col_idx]
        if rd is not None:
            img = _mask_for_plot(rd, rd["mid_idx"])
            ax.imshow(img, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(
                f"per_layer · t={t} · step {rd['mid_step']} · "
                f"J\u0303={rd['stability']['median']:.2f}",
                fontsize=10,
            )
            ax.set_yticks([0, rd["masks"][0].shape[0] - 1])
        else:
            ax.set_visible(False)
        if col_idx == 0:
            ax.set_ylabel("per_layer\nlayer idx", fontsize=11)

        # Row 2 — per_head (reference layer)
        rd = run_data.get(("per_head", t))
        ax = axes[2, col_idx]
        if rd is not None:
            img = _mask_for_plot(rd, rd["mid_idx"])
            ax.imshow(img, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(
                f"per_head · L{REF_LAYER} · t={t} · step {rd['mid_step']} · "
                f"J\u0303={rd['stability']['median']:.2f}",
                fontsize=10,
            )
            ax.set_yticks([0, img.shape[0] - 1])
        else:
            ax.set_visible(False)
        if col_idx == 0:
            ax.set_ylabel(f"per_head\nL{REF_LAYER} heads", fontsize=11)
        ax.set_xlabel("slot id")

    fig.suptitle(
        "Top-t slot selection — mid-training snapshot (Llama-3.2-3B, Qasper). "
        "Blue = selected. J\u0303 = median consecutive-snapshot Jaccard "
        "(mean across components) over all 19 logged snapshots.",
        fontsize=12,
        y=1.005,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path}")


def plot_per_head_full(run_data: dict, save_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 11))

    for col_idx, t in enumerate(TOP_TS):
        rd = run_data.get(("per_head", t))
        ax = axes[col_idx]
        if rd is None:
            ax.set_visible(False)
            continue

        mask = rd["masks"][rd["mid_idx"]]              # (L, H, N)
        n_layers, n_kv_heads, _ = mask.shape
        flat = mask.reshape(-1, mask.shape[-1])        # (L*H, N)

        ax.imshow(flat.float().numpy(), aspect="auto", cmap="Blues",
                  vmin=0, vmax=1, interpolation="nearest")
        for l in range(1, n_layers):
            ax.axhline(l * n_kv_heads - 0.5, color="red", lw=0.4, alpha=0.45)

        ax.set_title(
            f"per_head · t={t} · step {rd['mid_step']} · "
            f"J\u0303={rd['stability']['median']:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("slot id")
        ax.set_yticks(np.arange(n_layers) * n_kv_heads + (n_kv_heads - 1) / 2)
        ax.set_yticklabels([str(l) for l in range(n_layers)], fontsize=7)
        if col_idx == 0:
            ax.set_ylabel("layer idx  (8 heads stacked per layer)", fontsize=11)

    fig.suptitle(
        "Full per_head selection at mid-training (Llama-3.2-3B, 28 layers × 8 KV heads). "
        "Red lines separate adjacent layers.",
        fontsize=12,
        y=1.005,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path}")


def plot_three_batch_appendix(run_data: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    fractions = [0.25, 0.5, 0.75]

    for gran in GRANULARITIES:
        fig, axes = plt.subplots(
            len(TOP_TS), len(fractions),
            figsize=(15, 3.5 * len(TOP_TS)),
        )

        for row_idx, t in enumerate(TOP_TS):
            rd = run_data.get((gran, t))
            for col_idx, frac in enumerate(fractions):
                ax = axes[row_idx, col_idx]
                if rd is None:
                    ax.set_visible(False)
                    continue

                steps = rd["steps"]
                target = min(steps) + frac * (max(steps) - min(steps))
                idx = min(range(len(steps)), key=lambda i: abs(steps[i] - target))
                img = _mask_for_plot(rd, idx)

                ax.imshow(img, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.set_title(
                    f"t={t} · step {steps[idx]} (~{int(frac*100)}%)", fontsize=10,
                )
                if gran == "global":
                    ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"t={t}\n{gran}", fontsize=10)
                if row_idx == len(TOP_TS) - 1:
                    ax.set_xlabel("slot id")

        suptitle = (
            f"Appendix · {gran} · 3-batch small multiples at "
            f"~25 / 50 / 75 % of training"
        )
        if gran == "per_head":
            suptitle += f" (reference layer L{REF_LAYER})"
        fig.suptitle(suptitle, fontsize=12, y=1.002)
        plt.tight_layout()
        path = save_dir / f"{gran}_3batch.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {path}")


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------
def main():
    print(f"INVEST_DIR = {INVEST_DIR}")
    print(f"FIG_DIR    = {FIG_DIR}\n")

    run_data = load_all_runs()
    if not run_data:
        raise SystemExit("no runs loaded; check paths")

    add_mid_step(run_data)
    stability_stats(run_data)

    save_stability_table(run_data, INVEST_DIR / "stability_table.csv")
    save_stability_markdown(run_data, INVEST_DIR / "stability_table.md")

    # Also dump a compact JSON of the metadata (no tensors).
    meta = {
        f"{g}__t{t}": {
            "path": str(rd["path"]),
            "n_snapshots": len(rd["masks"]),
            "steps": rd["steps"],
            "mid_step": rd["mid_step"],
            "n_tokens": rd["n_tokens"],
            "n_layers": rd["n_layers"],
            "n_kv_heads": rd["n_kv_heads"],
            "stability": rd["stability"],
        }
        for (g, t), rd in run_data.items()
    }
    (INVEST_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[save] {INVEST_DIR / 'run_metadata.json'}")

    plot_main_snapshot(run_data, FIG_DIR / "main_snapshot_grid.png")
    plot_per_head_full(run_data, FIG_DIR / "per_head_full_grid.png")
    plot_three_batch_appendix(run_data, FIG_DIR / "appendix")

    print("\nDone.")


if __name__ == "__main__":
    main()
