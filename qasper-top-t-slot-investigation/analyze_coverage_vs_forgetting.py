"""
Link top-t selection coverage to Phase-1 forgetting (Llama, Qasper).

The mid-training snapshot analysis (analyze_slot_selection.py) showed that
finer granularity has *lower* step-to-step Jaccard.  That means finer
granularity rotates through *more* unique (layer, head, slot) triples over
training, so the union of all updated triples is larger.

Forgetting is determined by what *did* get written during Phase 2, not by what
Phase 2 stage-1 said was important.  So the right diagnostic for forgetting is
the cumulative coverage:

    coverage = | { (l, h, n) : (l, h, n) ever in top-t during Phase 2 } |
               -----------------------------------------------------------
                                 n_layers · n_kv_heads · n_tokens

We then:
  1. compute coverage per (granularity, top_t) using the 19 logged sparse-
     selection snapshots (every 30 optimizer steps),
  2. plot a coverage-growth curve (how the union grows as snapshots accumulate),
  3. plot a bar chart of final cumulative coverage by granularity × sparsity,
  4. scatter cumulative coverage against forgetting perplexity, and report
     Pearson + Spearman rank correlation.

All artifacts land in qasper-top-t-slot-investigation/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Re-use the loading helpers from the sibling script.
from analyze_slot_selection import (  # noqa: E402
    RUNS,
    GRANULARITIES,
    TOP_TS,
    find_slot_log,
    shapes_from_log,
    entry_to_mask,
)

REPO_ROOT = HERE.parent
PPL_CSV = (
    REPO_ROOT
    / "qasper-forgetting-investigation"
    / "plots"
    / "llama_granularity_x_sparsity_summary.csv"
)
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# CSV uses 'per-head' / 'per-layer' / 'global'; our keys use 'per_head' etc.
CSV_NAME = {"global": "global", "per_layer": "per-layer", "per_head": "per-head"}

GRAN_STYLE = {
    "global":    dict(color="#2ca02c", marker="^", label="global"),
    "per_layer": dict(color="#d6604d", marker="s", label="per_layer"),
    "per_head":  dict(color="#1f77b4", marker="o", label="per_head"),
}


# ----------------------------------------------------------------------
# Coverage helpers
# ----------------------------------------------------------------------
def expand_mask_to_lhn(
    mask: torch.Tensor, n_layers: int, n_kv_heads: int, n_tokens: int
) -> torch.Tensor:
    """Lift a granularity-specific mask up to a (L, H, N) boolean tensor.

    For the purposes of *forgetting*, the unit of analysis is the
    (layer, head, slot) triple, because that is exactly the granularity of
    `trainable_keys` / `trainable_values` cache parameters.
    """
    if mask.dim() == 1:                                       # global
        return mask.view(1, 1, -1).expand(n_layers, n_kv_heads, n_tokens).clone()
    if mask.dim() == 2:                                       # per_layer
        return mask.view(n_layers, 1, -1).expand(n_layers, n_kv_heads, n_tokens).clone()
    return mask.clone()                                       # per_head


def cumulative_coverage_growth(
    masks: list[torch.Tensor],
    n_layers: int,
    n_kv_heads: int,
    n_tokens: int,
) -> np.ndarray:
    """Return coverage fraction after including snapshots 1..k for k=1..K."""
    union = torch.zeros(n_layers, n_kv_heads, n_tokens, dtype=torch.bool)
    total = float(n_layers * n_kv_heads * n_tokens)
    fractions = []
    for m in masks:
        union |= expand_mask_to_lhn(m, n_layers, n_kv_heads, n_tokens)
        fractions.append(float(union.sum().item()) / total)
    return np.array(fractions)


# ----------------------------------------------------------------------
# Load forgetting perplexity table
# ----------------------------------------------------------------------
def load_forgetting_ppl() -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    with PPL_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["model"] != "llama":
                continue
            if row["top_k"] in ("baseline", "baseline_avg"):
                continue
            # Reverse-map 'per-head' -> 'per_head'.
            for our_key, csv_key in CSV_NAME.items():
                if row["granularity"] == csv_key:
                    out[(our_key, int(row["top_k"]))] = float(row["forgetting_ppl"])
                    break
    return out


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_coverage_growth(growth: dict, save_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for col_idx, t in enumerate(TOP_TS):
        ax = axes[col_idx]
        for gran in GRANULARITIES:
            key = (gran, t)
            if key not in growth:
                continue
            data = growth[key]
            xs = np.arange(1, len(data["fractions"]) + 1)
            style = GRAN_STYLE[gran]
            ax.plot(
                xs, data["fractions"],
                lw=2, ms=5, marker=style["marker"], color=style["color"],
                label=f"{gran} (final={data['fractions'][-1]:.2f})",
            )
        ax.axhline(t / data["n_tokens"], ls=":", color="grey", lw=1, alpha=0.7)
        ax.set_xlabel("# snapshots included (chronological)")
        if col_idx == 0:
            ax.set_ylabel("cumulative coverage =\n| ever-selected (l,h,n) | / (L·H·N)")
        ax.set_title(f"t = {t}", fontsize=11, weight="bold")
        ax.set_xticks([1, 5, 10, 15, 19])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.text(
            0.02, 0.96,
            f"per-step bound = t/N = {t}/{data['n_tokens']} = {t/data['n_tokens']:.2f}",
            transform=ax.transAxes,
            fontsize=8, va="top", color="grey", style="italic",
        )

    fig.suptitle(
        "Cumulative cache coverage during Phase 2 (sampled every ~30 steps; "
        "Llama-3.2-3B, Qasper). Dotted grey = per-step lower bound (t/N).",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path}")


def plot_coverage_bars(growth: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    n_grans = len(GRANULARITIES)
    bar_w = 0.25
    xs = np.arange(len(TOP_TS), dtype=float)

    for g_idx, gran in enumerate(GRANULARITIES):
        ys = [growth[(gran, t)]["fractions"][-1] for t in TOP_TS]
        ax.bar(
            xs + (g_idx - (n_grans - 1) / 2) * bar_w,
            ys,
            width=bar_w,
            color=GRAN_STYLE[gran]["color"],
            edgecolor="black",
            lw=0.5,
            label=gran,
        )
        for x_pos, y in zip(xs + (g_idx - (n_grans - 1) / 2) * bar_w, ys):
            ax.text(x_pos, y + 0.01, f"{y:.2f}", ha="center", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"t={t}" for t in TOP_TS])
    ax.set_ylabel("Cumulative coverage of (layer, head, slot) triples")
    ax.set_title(
        "Phase-2 cumulative cache coverage by granularity × sparsity\n"
        "(Llama-3.2-3B, union over 19 logged snapshots)",
        fontsize=12,
    )
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path}")


def plot_coverage_vs_ppl(
    growth: dict, ppl: dict[tuple[str, int], float], save_path: Path,
):
    """Scatter cumulative coverage vs forgetting perplexity."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    all_x, all_y = [], []
    for gran in GRANULARITIES:
        xs, ys, ts = [], [], []
        for t in TOP_TS:
            k = (gran, t)
            if k not in growth or k not in ppl:
                continue
            xs.append(growth[k]["fractions"][-1])
            ys.append(ppl[k])
            ts.append(t)
        style = GRAN_STYLE[gran]
        ax.plot(
            xs, ys,
            color=style["color"], lw=1.2, alpha=0.5, ls="--",
        )
        for x, y, t in zip(xs, ys, ts):
            ax.scatter(
                x, y,
                s=40 + (np.log2(t) - 4) * 60,  # size grows with t
                marker=style["marker"],
                color=style["color"],
                edgecolor="black",
                lw=0.6,
                zorder=5,
            )
            ax.annotate(
                f"t={t}",
                (x, y), textcoords="offset points", xytext=(7, -3),
                fontsize=8, color=style["color"],
            )
        all_x.extend(xs)
        all_y.extend(ys)

    # Fit a single line across all 12 points and report correlation.
    if len(all_x) >= 2:
        x = np.array(all_x)
        y = np.array(all_y)
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(min(x), max(x), 100)
        yy = np.polyval(coef, xx)
        ax.plot(xx, yy, color="grey", lw=1, alpha=0.7, label="linear fit")

        pearson = float(np.corrcoef(x, y)[0, 1])
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        spearman = float(np.corrcoef(rx, ry)[0, 1])
        ax.text(
            0.05, 0.95,
            f"Pearson r = {pearson:.3f}\nSpearman ρ = {spearman:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="grey", alpha=0.9),
        )

    handles = [
        plt.Line2D([], [], marker=GRAN_STYLE[g]["marker"],
                   color=GRAN_STYLE[g]["color"], lw=0, ms=8, label=g)
        for g in GRANULARITIES
    ]
    ax.legend(handles=handles, loc="lower right", title="granularity")
    ax.set_xlabel("Cumulative cache coverage during Phase 2  (fraction of L·H·N)")
    ax.set_ylabel("Forgetting — Phase 1 QA perplexity")
    ax.set_title(
        "Coverage of the cache during Phase 2 predicts Phase-1 forgetting\n"
        "(Llama-3.2-3B, Qasper; marker size grows with t)",
        fontsize=12,
    )
    ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path}")
    return pearson, spearman


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"HERE     = {HERE}")
    print(f"PPL_CSV  = {PPL_CSV}\n")

    growth: dict = {}
    for (gran, t), launch_dir in RUNS.items():
        path = find_slot_log(launch_dir)
        if path is None:
            print(f"[WARN] no sparse_slot_log.pt for ({gran}, t={t})")
            continue
        log = torch.load(path, weights_only=False)
        n_tokens, n_layers, n_kv_heads = shapes_from_log(log)
        masks = [entry_to_mask(e, n_tokens, n_layers, n_kv_heads) for e in log]
        fractions = cumulative_coverage_growth(masks, n_layers, n_kv_heads, n_tokens)
        growth[(gran, t)] = {
            "fractions": fractions,
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "n_snapshots": len(masks),
            "steps": [int(e["step"]) for e in log],
        }
        print(
            f"[load] {gran:<9} t={t:<3}  "
            f"final coverage = {fractions[-1]:.3f}  "
            f"per-step bound = {t/n_tokens:.3f}"
        )

    ppl = load_forgetting_ppl()
    print(f"\n[load] forgetting ppl entries: {len(ppl)}")

    # Combined CSV: coverage + ppl per (granularity, t).
    csv_path = HERE / "coverage_vs_forgetting.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow([
            "granularity", "top_t", "n_snapshots", "n_tokens",
            "per_step_bound", "cumulative_coverage",
            "forgetting_qa_ppl",
        ])
        for k in sorted(growth.keys(), key=lambda x: (GRANULARITIES.index(x[0]), x[1])):
            rd = growth[k]
            w.writerow([
                k[0], k[1], rd["n_snapshots"], rd["n_tokens"],
                f"{k[1]/rd['n_tokens']:.4f}",
                f"{rd['fractions'][-1]:.4f}",
                f"{ppl.get(k, float('nan')):.4f}",
            ])
    print(f"[save] {csv_path}")

    plot_coverage_growth(growth, FIG_DIR / "coverage_growth.png")
    plot_coverage_bars(growth, FIG_DIR / "coverage_bars.png")
    pearson, spearman = plot_coverage_vs_ppl(
        growth, ppl, FIG_DIR / "coverage_vs_ppl.png"
    )

    print("\n=== Pearson / Spearman across all 12 (granularity, t) ===")
    print(f"  Pearson  r = {pearson:.3f}")
    print(f"  Spearman ρ = {spearman:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
