"""
Granularity overlap analysis for sparse-cache cartridges (qasper).

Measures how much information is lost by aggregating fine-grained
attention/TF-IDF rankings up the hierarchy [global -> per-layer -> per-head]
using **Jaccard similarity against a consensus ranking**.

Stage 1 (background stats / extra-batch dynamics, top-k = 128):
    Figure 1A:  How much each *layer*'s top-128 disagrees with a cross-layer
                consensus (Borda) -- shows how lossy the global aggregation is.
    Figure 1B:  Within each layer, how much each *head*'s top-128 disagrees
                with a within-layer cross-head consensus -- shows how lossy
                the per-layer aggregation is.

Stage 2 (intra-batch TF-IDF dynamics, t in [64, 128, 256, 512]):
    Figure 2A:  For each layer, mean Jaccard(per-head top-t, per-layer-
                aggregated top-t), with one line per t -- shows how much each
                head deviates from its parent layer as sparsity changes.
    Figure 2B:  For each layer, mean pairwise Jaccard among heads' top-t,
                with one line per t -- shows how spread the heads are inside
                a layer as sparsity changes.

Default setting: qasper, batch size 32. Stage 2 per_head TF logs are only
available at batch size 64 (no per_head bsize=32 stage-2 runs were trained);
this is annotated in Figure 2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths (qasper)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

PER_LAYER_BG_STATS = (
    OUTPUTS
    / "qasper-initial-per-layer-all-reduce"
    / "d8103e75-4886-47a0-8af0-286ca4bec665"
    / "bg_stats.pt"
)
PER_HEAD_BG_STATS = (
    OUTPUTS
    / "qasper-initial-per-head-all-reduce"
    / "51f1e2fb-321a-4cee-8796-9a826fb6681d"
    / "bg_stats.pt"
)

PER_HEAD_TFIDF_LOG = (
    OUTPUTS
    / "2026-06-18-04-59-17-continual_sparse-qasper-per-head-top-128-value-only_all-reduce-bsize-64"
    / "f8e38da6-e293-4246-a873-f80f84b55d65"
    / "tfidf_ranking_log.pt"
)

FIG_DIR = REPO_ROOT / "notebook" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_STAGE1 = 128
TS_STAGE2 = [64, 128, 256, 512]
REFERENCE_LAYER = 14  # the middle layer used for the per-head distribution panel


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def topk_indices_to_mask(top_idx: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Convert a top-k index tensor (..., k) to a boolean (..., n_tokens) mask."""
    shape = top_idx.shape[:-1] + (n_tokens,)
    mask = torch.zeros(shape, dtype=torch.bool, device=top_idx.device)
    mask.scatter_(-1, top_idx, True)
    return mask


def jaccard_bool(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Jaccard between two broadcastable boolean (..., n_tokens) sets along the last dim."""
    inter = (a & b).sum(-1).float()
    union = (a | b).sum(-1).float()
    return torch.where(union > 0, inter / union, torch.ones_like(inter))


def borda_consensus_topk(
    ranked_positions: torch.Tensor,
    sum_dim: int,
    k: int,
) -> torch.Tensor:
    """Borda-count consensus top-k aggregated over `sum_dim`.

    Args:
        ranked_positions: long tensor of position indices, sorted descending by
            score along the last dim. Shape (..., n_units, n_tokens) where
            n_units is the dim being aggregated.
        sum_dim: the axis whose entries are aggregated into the consensus.
        k: top-k to return.

    Returns:
        Long tensor (..., k) of consensus top-k positions.
    """
    n_tokens = ranked_positions.shape[-1]
    inverse_rank = torch.zeros_like(ranked_positions)
    arange = torch.arange(n_tokens, device=ranked_positions.device).expand_as(
        ranked_positions
    )
    inverse_rank.scatter_(-1, ranked_positions, arange)
    score = (n_tokens - 1 - inverse_rank).float()
    consensus_score = score.sum(dim=sum_dim)
    _, consensus_top = torch.topk(consensus_score, k, dim=-1)
    return consensus_top


def pairwise_jaccard_within(mask: torch.Tensor) -> torch.Tensor:
    """Pairwise Jaccard among the 2nd-to-last axis of a (..., m, n_tokens) bool mask.

    Returns a (..., m, m) tensor with diagonal = 1.
    """
    fm = mask.float()
    inter = fm @ fm.transpose(-1, -2)
    sums = fm.sum(-1, keepdim=True)
    union = sums + sums.transpose(-1, -2) - inter
    return torch.where(union > 0, inter / union, torch.ones_like(inter))


# ---------------------------------------------------------------------------
# Stage 1 -- extra-batch dynamics
# ---------------------------------------------------------------------------
def stage1_layer_vs_consensus(per_layer_bg_path: Path, top_k: int) -> dict:
    """Per-batch Jaccard(layer top-k, cross-layer consensus top-k)."""
    print(f"[stage1] loading per_layer bg_stats: {per_layer_bg_path}")
    data = torch.load(per_layer_bg_path, weights_only=False, map_location="cpu")
    ranked = data["batch_ranked_positions"]  # (n_batches, n_layers, n_tokens)
    n_batches, n_layers, n_tokens = ranked.shape
    print(f"  shape: ({n_batches}, {n_layers}, {n_tokens})")

    layer_top = ranked[..., :top_k]  # (n_batches, n_layers, k)
    layer_mask = topk_indices_to_mask(layer_top, n_tokens)

    consensus_top = borda_consensus_topk(ranked, sum_dim=1, k=top_k)
    consensus_mask = topk_indices_to_mask(consensus_top, n_tokens)

    jacc = jaccard_bool(layer_mask, consensus_mask.unsqueeze(1))  # (n_batches, n_layers)
    return {
        "jaccard": jacc.numpy(),
        "n_batches": n_batches,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
    }


def stage1_head_vs_consensus(per_head_bg_path: Path, top_k: int) -> dict:
    """Per-batch Jaccard(head top-k, within-layer head consensus top-k)."""
    print(f"[stage1] loading per_head bg_stats: {per_head_bg_path}")
    data = torch.load(per_head_bg_path, weights_only=False, map_location="cpu")
    ranked = data["batch_ranked_positions"]  # (n_batches, n_layers, n_heads, n_tokens)
    n_batches, n_layers, n_heads, n_tokens = ranked.shape
    print(f"  shape: ({n_batches}, {n_layers}, {n_heads}, {n_tokens})")

    head_top = ranked[..., :top_k]  # (n_b, n_l, n_h, k)
    head_mask = topk_indices_to_mask(head_top, n_tokens)

    consensus_top = borda_consensus_topk(ranked, sum_dim=2, k=top_k)
    consensus_mask = topk_indices_to_mask(consensus_top, n_tokens)

    jacc = jaccard_bool(head_mask, consensus_mask.unsqueeze(2))  # (n_b, n_l, n_h)

    pairwise = pairwise_jaccard_within(head_mask)  # (n_b, n_l, n_h, n_h)
    eye = torch.eye(n_heads, dtype=torch.bool)
    off_diag = pairwise[..., ~eye].view(n_batches, n_layers, -1)
    pairwise_per_layer = off_diag.mean(dim=-1).numpy()  # (n_b, n_l)

    return {
        "jaccard_head_vs_consensus": jacc.numpy(),
        "pairwise_head_jaccard": pairwise_per_layer,
        "n_batches": n_batches,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_tokens": n_tokens,
    }


def plot_stage1(layer_stats: dict, head_stats: dict, top_k: int, save_path: Path):
    n_layers = layer_stats["n_layers"]
    n_heads = head_stats["n_heads"]
    layer_jacc = layer_stats["jaccard"]                       # (n_b, n_l)
    head_jacc = head_stats["jaccard_head_vs_consensus"]       # (n_b, n_l, n_h)
    pair_jacc = head_stats["pairwise_head_jaccard"]           # (n_b, n_l)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    ax = axes[0]
    ax.boxplot(
        [layer_jacc[:, l] for l in range(n_layers)],
        positions=np.arange(n_layers),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#9ec5fe", edgecolor="#1f4e8e"),
        medianprops=dict(color="#1f4e8e"),
    )
    layer_mean = layer_jacc.mean(axis=0)
    ax.plot(
        np.arange(n_layers),
        layer_mean,
        "o-",
        color="#1f4e8e",
        label=f"per-batch mean (overall = {layer_jacc.mean():.3f})",
    )
    ax.axhline(layer_jacc.mean(), color="#1f4e8e", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel(f"Jaccard(layer top-{top_k}, cross-layer consensus)")
    ax.set_title(
        f"(A) Global vs per-layer — how much each layer disagrees with the "
        f"cross-layer consensus (top-{top_k}, n_batches={layer_stats['n_batches']})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    flat_per_layer = head_jacc.transpose(1, 0, 2).reshape(n_layers, -1)
    ax.boxplot(
        [flat_per_layer[l] for l in range(n_layers)],
        positions=np.arange(n_layers),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#fcb6b6", edgecolor="#9c1c1c"),
        medianprops=dict(color="#9c1c1c"),
    )
    head_mean = head_jacc.mean(axis=(0, 2))
    ax.plot(
        np.arange(n_layers),
        head_mean,
        "o-",
        color="#9c1c1c",
        label=f"head vs head-consensus (overall = {head_jacc.mean():.3f})",
    )
    pair_mean = pair_jacc.mean(axis=0)
    ax.plot(
        np.arange(n_layers),
        pair_mean,
        "s--",
        color="#5a0a0a",
        label=f"mean pairwise head Jaccard (overall = {pair_jacc.mean():.3f})",
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"Jaccard(head top-{top_k}, within-layer head consensus)")
    ax.set_title(
        f"(B) Per-layer vs per-head — how much each head disagrees with its own "
        f"layer's head consensus ({n_heads} heads/layer, top-{top_k})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    ax.set_xlim(-0.5, n_layers - 0.5)

    fig.suptitle(
        "Stage 1 (background stats, qasper, bsize=32) — "
        "information lost when aggregating up the hierarchy",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_stage1_reference_layer(
    head_stats: dict, top_k: int, layer_idx: int, save_path: Path
):
    """Detailed per-head view inside a single reference layer."""
    head_jacc = head_stats["jaccard_head_vs_consensus"]  # (n_b, n_l, n_h)
    n_heads = head_stats["n_heads"]
    pair_jacc = head_stats["pairwise_head_jaccard"]       # (n_b, n_l)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    data_per_head = [head_jacc[:, layer_idx, h] for h in range(n_heads)]
    bp = ax.boxplot(
        data_per_head,
        positions=np.arange(n_heads),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#fcb6b6", edgecolor="#9c1c1c"),
        medianprops=dict(color="#9c1c1c"),
    )
    head_means = head_jacc[:, layer_idx, :].mean(axis=0)
    ax.plot(np.arange(n_heads), head_means, "o-", color="#9c1c1c", label="head mean")
    ax.axhline(
        pair_jacc[:, layer_idx].mean(),
        color="#1f4e8e",
        ls="--",
        label=f"mean pairwise (= {pair_jacc[:, layer_idx].mean():.3f})",
    )
    ax.set_xticks(np.arange(n_heads))
    ax.set_xlabel("Head index (within layer)")
    ax.set_ylabel(f"Jaccard(head top-{top_k}, head consensus)")
    ax.set_title(f"Stage 1 reference layer = {layer_idx}: head spread vs consensus")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Stage 2 -- intra-batch dynamics over t in [64, 128, 256, 512]
# ---------------------------------------------------------------------------
def stage2_head_vs_layer_aggregate(
    per_head_tfidf_path: Path, t_values: list[int]
) -> dict:
    """For each step in the per_head stage-2 log, compute Jaccard between
    per-head top-t and per-layer-aggregated top-t, plus pairwise head Jaccard.

    Returns aggregated arrays of shape (n_steps, n_layers, n_heads) and
    (n_steps, n_layers) per t.
    """
    print(f"[stage2] loading per_head tfidf log: {per_head_tfidf_path}")
    data = torch.load(per_head_tfidf_path, weights_only=False, map_location="cpu")
    info_list = data["per_step_info"]
    granularity = data["granularity"]
    assert granularity == "per_head", f"need per_head TF, got {granularity}"

    n_steps = len(info_list)
    sample_tfidf = info_list[0]["tfidf"]  # (n_layers, n_heads, n_tokens)
    n_layers, n_heads, n_tokens = sample_tfidf.shape
    print(
        f"  n_steps={n_steps}, n_layers={n_layers}, n_heads={n_heads}, n_tokens={n_tokens}"
    )

    out: dict[int, dict] = {}
    for t in t_values:
        out[t] = {
            "head_vs_layer": np.zeros((n_steps, n_layers, n_heads)),
            "pairwise_head": np.zeros((n_steps, n_layers)),
        }

    for s_idx, info in enumerate(info_list):
        tfidf = info["tfidf"].float()  # (L, H, T)
        layer_score = tfidf.sum(dim=1)  # (L, T)

        for t in t_values:
            t_eff = min(t, n_tokens)
            _, head_top = torch.topk(tfidf, t_eff, dim=-1)        # (L, H, t)
            head_mask = topk_indices_to_mask(head_top, n_tokens)
            _, layer_top = torch.topk(layer_score, t_eff, dim=-1) # (L, t)
            layer_mask = topk_indices_to_mask(layer_top, n_tokens)

            j_head_layer = jaccard_bool(head_mask, layer_mask.unsqueeze(1))  # (L, H)
            out[t]["head_vs_layer"][s_idx] = j_head_layer.numpy()

            pairwise = pairwise_jaccard_within(head_mask)  # (L, H, H)
            eye = torch.eye(n_heads, dtype=torch.bool)
            mean_pair_per_layer = pairwise[..., ~eye].view(n_layers, -1).mean(dim=-1)
            out[t]["pairwise_head"][s_idx] = mean_pair_per_layer.numpy()

    return {
        "per_t": out,
        "n_steps": n_steps,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_tokens": n_tokens,
        "steps": [info["step"] for info in info_list],
        "training_top_t": info_list[0]["top_t"],
    }


def plot_stage2(stage2: dict, t_values: list[int], save_path: Path):
    n_layers = stage2["n_layers"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(t_values) - 1)) for i in range(len(t_values))]

    ax = axes[0]
    for t, color in zip(t_values, colors):
        arr = stage2["per_t"][t]["head_vs_layer"]  # (n_steps, n_layers, n_heads)
        mean_per_layer = arr.mean(axis=(0, 2))
        std_per_layer = arr.std(axis=(0, 2))
        ax.plot(
            np.arange(n_layers),
            mean_per_layer,
            "o-",
            color=color,
            label=f"t = {t}  (overall = {arr.mean():.3f})",
        )
        ax.fill_between(
            np.arange(n_layers),
            mean_per_layer - std_per_layer,
            mean_per_layer + std_per_layer,
            color=color,
            alpha=0.15,
        )
    ax.set_ylabel("Jaccard(head top-t, per-layer aggregated top-t)")
    ax.set_title(
        "(A) Per-layer vs per-head — how much each head disagrees with its layer's aggregated ranking "
        "(higher = better agreement)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", title="sparsity t")
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    for t, color in zip(t_values, colors):
        arr = stage2["per_t"][t]["pairwise_head"]  # (n_steps, n_layers)
        mean_per_layer = arr.mean(axis=0)
        std_per_layer = arr.std(axis=0)
        ax.plot(
            np.arange(n_layers),
            mean_per_layer,
            "o-",
            color=color,
            label=f"t = {t}  (overall = {arr.mean():.3f})",
        )
        ax.fill_between(
            np.arange(n_layers),
            mean_per_layer - std_per_layer,
            mean_per_layer + std_per_layer,
            color=color,
            alpha=0.15,
        )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean pairwise Jaccard among heads' top-t")
    ax.set_title(
        "(B) Within-layer head spread — pairwise agreement among heads "
        "(lower = heads disagree more)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", title="sparsity t")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    ax.set_xlim(-0.5, n_layers - 0.5)

    src_run = "per_head top_t=128 (bsize=64; only batch size with per_head TF logs)"
    fig.suptitle(
        "Stage 2 (TF-IDF intra-batch dynamics, qasper) — varying sparsity t\n"
        f"source: {src_run}; "
        f"{stage2['n_steps']} logged steps × {stage2['n_heads']} heads/layer",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Stage 1 (background stats, qasper, bsize=32, top-k = %d)" % TOP_K_STAGE1)
    print("=" * 72)
    layer_stats = stage1_layer_vs_consensus(PER_LAYER_BG_STATS, TOP_K_STAGE1)
    head_stats = stage1_head_vs_consensus(PER_HEAD_BG_STATS, TOP_K_STAGE1)

    print(
        f"  per-layer Jaccard vs cross-layer consensus: "
        f"mean = {layer_stats['jaccard'].mean():.3f}, "
        f"std = {layer_stats['jaccard'].std():.3f}"
    )
    print(
        f"  per-head Jaccard vs within-layer head consensus: "
        f"mean = {head_stats['jaccard_head_vs_consensus'].mean():.3f}, "
        f"std = {head_stats['jaccard_head_vs_consensus'].std():.3f}"
    )
    print(
        f"  pairwise head Jaccard within layer:           "
        f"mean = {head_stats['pairwise_head_jaccard'].mean():.3f}, "
        f"std = {head_stats['pairwise_head_jaccard'].std():.3f}"
    )

    plot_stage1(
        layer_stats,
        head_stats,
        TOP_K_STAGE1,
        FIG_DIR / "stage1_granularity_jaccard.png",
    )
    plot_stage1_reference_layer(
        head_stats,
        TOP_K_STAGE1,
        REFERENCE_LAYER,
        FIG_DIR / f"stage1_reference_layer{REFERENCE_LAYER}_heads.png",
    )

    print("\n" + "=" * 72)
    print("Stage 2 (TF-IDF intra-batch, t in %s)" % TS_STAGE2)
    print("=" * 72)
    stage2 = stage2_head_vs_layer_aggregate(PER_HEAD_TFIDF_LOG, TS_STAGE2)
    print("  Summary (mean across all steps × layers × heads):")
    for t in TS_STAGE2:
        m_hl = stage2["per_t"][t]["head_vs_layer"].mean()
        m_pp = stage2["per_t"][t]["pairwise_head"].mean()
        print(
            f"    t={t:>4}: head-vs-layer-aggregate Jaccard = {m_hl:.3f} | "
            f"pairwise head Jaccard = {m_pp:.3f}"
        )

    plot_stage2(
        stage2,
        TS_STAGE2,
        FIG_DIR / "stage2_granularity_jaccard.png",
    )

    print("\nDone. Figures written to:")
    for p in sorted(FIG_DIR.glob("stage*_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
