"""Shared primitives for the plot_*.py scripts.

Convention: each experiment family has ONE plot script (plot_baselines.py,
plot_per_layer_swap.py, plot_per_slot_swap.py, plot_per_question_heatmap.py).
Adding a new view means adding a flag to one of these scripts, NOT a new
file. Only create a new plot_*.py when the input data type or panel
layout is fundamentally different.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def load_baseline_json(base_dir: Path, mode: str, which: str,
                       eval_name: str = "amd_2021") -> dict:
    """Read `<base_dir>/<mode>/<which>.json` and unwrap the `<eval_name>`
    top-level key if present (per_layer_swap.py nests per eval).

    mode ∈ {"openended", "yesno", "mcq"}; which ∈ {"trained", "untrained"} (or
    legacy "baseline_A"/"baseline_B").
    """
    path = base_dir / mode / f"{which}.json"
    with open(path) as fh:
        d = json.load(fh)
    return d[eval_name] if eval_name in d else d


def load_layer_swap_summary(summary_path: Path, eval_name: str,
                            direction: str = "A_into_B",
                            baseline_a: str = "baseline_A",
                            baseline_b: str = "baseline_B",
                            metric: str = "token_nll") -> dict:
    """Pull the phase-1/phase-2 baselines and the per-layer metric curve out
    of a per_layer_swap.py `summary.json`. Returns
    `{"nll_A", "nll_B", "layers": [...], "values": [...]}`.
    """
    s = json.loads(Path(summary_path).read_text())
    nll_A = s[baseline_a][eval_name][metric]
    nll_B = s[baseline_b][eval_name][metric]
    pat = re.compile(rf"{direction}_L(\d+)$")
    rows = []
    for k, v in s.items():
        m = pat.match(k)
        if m:
            rows.append((int(m.group(1)), v[eval_name][metric]))
    rows.sort()
    return {
        "nll_A": nll_A,
        "nll_B": nll_B,
        "layers": [r[0] for r in rows],
        "values": [r[1] for r in rows],
    }


def load_per_slot_pt(pt_path: Path) -> dict:
    """Unwrap a per_slot_swap.pt bundle (written by per_slot_swap.py):
    `{"R", "base_nll", "donor_nll", "log_ppl_per_slot"}`. Returns a dict
    with numpy R and the two scalar baselines.
    """
    import torch
    d = torch.load(Path(pt_path))
    return {
        "R": d["R"].numpy(),
        "R_tensor": d["R"],
        "base_nll": float(d["base_nll"]),
        "donor_nll": float(d["donor_nll"]),
        "log_ppl_per_slot": d.get("log_ppl_per_slot"),
    }


# -----------------------------------------------------------------------------
# Grid-config iterators
# -----------------------------------------------------------------------------

def iter_configs(models: Iterable[str], tokens: Iterable[int]) -> Iterator[
    tuple[int, int, str, int]
]:
    """Yield (row, col, model_short, toks) for (model, toks) grid plots."""
    models = list(models)
    tokens = list(tokens)
    for r, model in enumerate(models):
        for c, toks in enumerate(tokens):
            yield r, c, model, toks


def get_axes(axes, r: int, c: int, nrows: int, ncols: int):
    """Robustly index into matplotlib axes arrays regardless of 1D/2D layout."""
    if nrows == 1 and ncols == 1:
        return axes
    if nrows == 1:
        return axes[c]
    if ncols == 1:
        return axes[r]
    return axes[r][c]


# -----------------------------------------------------------------------------
# Panel helpers
# -----------------------------------------------------------------------------

def bar_panel_trained_vs_untrained(
    ax,
    rows: list[dict],
    key_t: str,
    key_u: str,
    title: str,
    is_acc: bool,
    chance_line: float | None = None,
    legend_fontsize: int = 7,
) -> None:
    """Draw a two-bar-per-x panel: trained (blue) vs untrained (red).

    `rows` is a list of dicts each containing at least {"toks": int, key_t:
    float, key_u: float}. If `is_acc`, values are formatted as %.
    """
    xs = np.arange(len(rows))
    w = 0.36
    tv = [r[key_t] for r in rows]
    uv = [r[key_u] for r in rows]
    b1 = ax.bar(xs - w / 2, tv, w, label="trained", color="#1f77b4")
    b2 = ax.bar(xs + w / 2, uv, w, label="untrained (step-0)", color="#d62728")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"toks={r['toks']}" for r in rows])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=legend_fontsize)
    if is_acc:
        ax.set_ylim(0, 1.0)
        if chance_line is not None:
            ax.axhline(chance_line, color="gray", lw=0.7, ls="--")
        fmt = lambda v: f"{v * 100:.1f}%"
    else:
        fmt = lambda v: f"{v:.2f}"
    for bars, vals in [(b1, tv), (b2, uv)]:
        for bar, v in zip(bars, vals):
            ax.annotate(
                fmt(v),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7,
            )


def curve_panel_layer_swap(
    ax,
    curve: dict,
    panel_title: str,
    swap_color: str = "#d62728",
    base_color: str = "#222222",
    donor_color: str = "#2ca02c",
    swap_label: str = "per-layer swap",
    base_label: str = "base baseline",
    donor_label: str = "donor baseline",
    xlabel: str = "Layer index L (V at L replaced with donor's V)",
    ylabel: str = "log-perplexity",
) -> None:
    """Plot one per-layer line from load_layer_swap_summary() output."""
    ax.plot(curve["layers"], curve["values"], "-", color=swap_color, lw=1.1,
            label=swap_label)
    ax.axhline(curve["nll_B"], color=base_color, ls="--", lw=1,
               label=f"{base_label}: {curve['nll_B']:.3f}")
    ax.axhline(curve["nll_A"], color=donor_color, ls="--", lw=1,
               label=f"{donor_label}: {curve['nll_A']:.3f}")
    ax.set_title(panel_title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="center right", fontsize=7)
    ax.grid(alpha=0.25)


def bar_panel_per_slot_R(
    ax,
    slot: dict,
    panel_title: str,
    top_k_label: int = 5,
    pos_color: str = "#d62728",
    neg_color: str = "#1f77b4",
    show_full_restoration_line: bool = True,
    annotate_prefix: str = "",
) -> None:
    """Plot R(p) across slots (one bar per slot, color = sign) plus top-k
    annotations. `slot` is from load_per_slot_pt()."""
    import torch
    R = slot["R"]
    n = len(R)
    colors = [pos_color if v > 0 else neg_color for v in R]
    ax.bar(range(n), R, color=colors, width=1.0)
    ax.axhline(0, color="k", lw=0.6)
    if show_full_restoration_line:
        ax.axhline(1.0, color="#2ca02c", ls="--", lw=1,
                   label="full restoration (R=1)")
    ax.set_xlabel("slot position p")
    ax.set_ylabel("restoration R(p)")
    ax.set_title(panel_title, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    if top_k_label > 0:
        top_idx = torch.topk(slot["R_tensor"], top_k_label).indices.tolist()
        for p in top_idx:
            ax.annotate(f"{annotate_prefix}{p}", (p, R[p]),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=6)


def line_panel_per_slot_nll(
    ax,
    slot: dict,
    panel_title: str,
    top_k_label: int = 5,
    swap_color: str = "#d62728",
    base_color: str = "#222222",
    donor_color: str = "#2ca02c",
    swap_label: str = "per-slot swap",
    base_label: str = "untrained (base)",
    donor_label: str = "trained (donor)",
    annotate_prefix: str = "",
) -> None:
    """Plot raw log-perplexity per swapped slot as a line, with base and
    donor baselines as dashed references. `slot` is from load_per_slot_pt().
    Annotates the top-K slots with the LOWEST swap nll (best restoration)."""
    lp = slot.get("log_ppl_per_slot")
    if lp is None:
        raise ValueError("log_ppl_per_slot missing from slot bundle")
    import torch
    if isinstance(lp, torch.Tensor):
        lp_np = lp.numpy()
    else:
        lp_np = np.asarray(lp)
    n = len(lp_np)
    ax.plot(range(n), lp_np, "-", color=swap_color, lw=0.9, label=swap_label)
    ax.axhline(slot["base_nll"], color=base_color, ls="--", lw=1,
               label=f"{base_label}: {slot['base_nll']:.3f}")
    ax.axhline(slot["donor_nll"], color=donor_color, ls="--", lw=1,
               label=f"{donor_label}: {slot['donor_nll']:.3f}")
    ax.set_xlabel("slot position p")
    ax.set_ylabel("log-perplexity")
    ax.set_title(panel_title, fontsize=10)
    ax.grid(alpha=0.25)
    if top_k_label > 0:
        lp_tensor = lp if isinstance(lp, torch.Tensor) else torch.tensor(lp_np)
        top_idx = torch.topk(-lp_tensor, top_k_label).indices.tolist()
        for p in top_idx:
            ax.annotate(f"{annotate_prefix}{p}", (p, lp_np[p]),
                        textcoords="offset points", xytext=(0, -10),
                        ha="center", fontsize=6)


GROUP_COLOR = {
    "top3":  "#fdae61",
    "top5":  "#f46d43",
    "top10": "#d73027",
    "bottom_third": "#91bfdb",
    "middle_third": "#4575b4",
    "top_third":    "#313695",
    "mid_half":     "#1a9850",
}


def _layer_swap_groups_baselines(summary, eval_name, direction, a_key, b_key):
    if direction == "A_into_B":
        base_key, donor_key = b_key, a_key
    else:
        base_key, donor_key = a_key, b_key
    return (
        summary[base_key][eval_name]["token_nll"],
        summary[donor_key][eval_name]["token_nll"],
    )


def bar_panel_layer_swap_groups_raw(
    ax, summary, eval_name, direction, groups, panel_title,
    a_key, b_key, base_label, donor_label,
) -> None:
    """Bars of raw token_nll per layer-group + base/donor baseline lines."""
    base_nll, donor_nll = _layer_swap_groups_baselines(
        summary, eval_name, direction, a_key, b_key,
    )
    nlls = [summary[f"{direction}_G_{g}"][eval_name]["token_nll"] for g in groups]
    colors = [GROUP_COLOR.get(g, "#888888") for g in groups]
    ax.bar(range(len(groups)), nlls, color=colors, width=0.7)
    ax.axhline(base_nll, color="#222222", ls="--", lw=1,
               label=f"{base_label}: {base_nll:.3f}")
    ax.axhline(donor_nll, color="#2ca02c", ls="--", lw=1,
               label=f"{donor_label}: {donor_nll:.3f}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=0, fontsize=9)
    ax.set_ylabel("log-perplexity (token_nll)")
    ax.set_title(panel_title, fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(nlls):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7)


def bar_panel_layer_swap_groups_R(
    ax, summary, eval_name, direction, groups, panel_title,
    a_key, b_key,
) -> None:
    """Bars of restoration R=(nll_base-nll_swap)/(nll_base-nll_donor)."""
    base_nll, donor_nll = _layer_swap_groups_baselines(
        summary, eval_name, direction, a_key, b_key,
    )
    gap = base_nll - donor_nll
    Rs = []
    for g in groups:
        n = summary[f"{direction}_G_{g}"][eval_name]["token_nll"]
        Rs.append((base_nll - n) / gap if gap != 0 else 0.0)
    colors = [GROUP_COLOR.get(g, "#888888") for g in groups]
    ax.bar(range(len(groups)), Rs, color=colors, width=0.7)
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1, label="target (R=1)")
    ax.axhline(0.0, color="#222222", ls="--", lw=1, label="base (R=0)")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=0, fontsize=9)
    ax.set_ylabel("restoration R")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title(panel_title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    for i, R in enumerate(Rs):
        ax.text(i, R + 0.02, f"{R:.2f}", ha="center", fontsize=7)


