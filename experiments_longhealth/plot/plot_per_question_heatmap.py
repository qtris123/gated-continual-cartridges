#!/usr/bin/env python3
"""Per-question heatmap over layers or slot positions.

One script for two views of a V-swap sweep, selected by --dimension:
  layer:  reads per-layer JSONs in a per_layer_swap output dir.
  slot :  reads a per_slot_swap.pt bundle written with --per-question.

    # Layer heatmap
    plot_per_question_heatmap.py --dimension layer \\
      --input outputs/layer_swap_qwen3_toks1024_init_vs_trained_both \\
      --direction A_into_B --output out.png

    # Slot heatmap
    plot_per_question_heatmap.py --dimension slot \\
      --input outputs/per_slot_qwen3_toks1024_.../per_slot_swap.pt \\
      --output out.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TYPE_ORDER = ["factual", "knowledge_K", "synthesize_S",
              "structure_ST", "reasoning_R", "multi_hop"]


def _load_per_question(path: Path, eval_name: str) -> dict[str, float]:
    d = json.load(open(path))[eval_name]["results"]
    return {r["question_id"]: r["mean_nll"] for r in d}


def _load_layer(run_dir: Path, eval_name: str, direction: str,
                baseline_a: str, baseline_b: str):
    pat = re.compile(rf"{direction}_L(\d+)\.json$")
    layers_files = sorted(
        ((int(pat.search(p.name).group(1)), p)
         for p in run_dir.iterdir() if pat.search(p.name)),
        key=lambda x: x[0],
    )
    if not layers_files:
        raise SystemExit(f"No {direction}_L*.json in {run_dir}")
    cols = [L for L, _ in layers_files]

    a_map = _load_per_question(run_dir / f"{baseline_a}.json", eval_name)
    b_map = _load_per_question(run_dir / f"{baseline_b}.json", eval_name)
    qids = sorted(set(a_map) & set(b_map))

    M = np.zeros((len(qids), len(cols)), dtype=float)
    for c, (_, p) in enumerate(layers_files):
        d = _load_per_question(p, eval_name)
        for r, q in enumerate(qids):
            M[r, c] = d[q]

    a_vec = np.array([a_map[q] for q in qids])
    b_vec = np.array([b_map[q] for q in qids])
    return M, a_vec, b_vec, qids, cols, "Layer L swapped"


def _load_slot(pt_path: Path, baseline_a: str, baseline_b: str):
    import torch
    bundle = torch.load(pt_path, weights_only=False)
    for req in ("grid_per_q", "base_per_q", "donor_per_q", "question_ids"):
        if req not in bundle:
            raise SystemExit(f"{pt_path} missing '{req}' — "
                             "re-run per_slot_swap.py with --per-question")
    M = bundle["grid_per_q"].numpy()
    a_vec = bundle["donor_per_q"].numpy()  # donor = A (typically trained)
    b_vec = bundle["base_per_q"].numpy()   # base  = B (typically untrained)
    qids = list(bundle["question_ids"])

    alpha = np.argsort(qids)
    qids = [qids[i] for i in alpha]
    return (
        M[alpha], a_vec[alpha], b_vec[alpha],
        qids, list(range(M.shape[1])), "slot position p",
    )


def _direction_sides(direction: str, a_vec: np.ndarray, b_vec: np.ndarray,
                     baseline_a: str, baseline_b: str):
    if direction == "A_into_B":
        return b_vec, a_vec, baseline_b, baseline_a  # base, donor, base_name, donor_name
    return a_vec, b_vec, baseline_a, baseline_b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dimension", required=True, choices=["layer", "slot"])
    ap.add_argument("--input", required=True,
                    help="Layer: run directory with per-layer JSONs. "
                         "Slot : path to per_slot_swap.pt bundle.")
    ap.add_argument("--eval-name", default="amd_2021",
                    help="(Layer only) eval key inside the per-layer JSONs.")
    ap.add_argument("--direction", default="A_into_B",
                    choices=["A_into_B", "B_into_A"])
    ap.add_argument("--baseline-a-label", default="trained",
                    help="JSON filename (layer) / donor label (slot).")
    ap.add_argument("--baseline-b-label", default="untrained",
                    help="JSON filename (layer) / base label (slot).")
    ap.add_argument("--sort-by", default="none",
                    choices=["trained", "untrained", "gap", "none"],
                    help="'none' keeps alphabetical qid order (default).")
    ap.add_argument("--metric", default="nll", choices=["nll", "delta_base"])
    ap.add_argument("--group-by-type", action="store_true")
    ap.add_argument("--clip-percentile", type=float, default=100.0,
                    help="delta_base: clip |range| at this percentile. "
                         "nll: clip both tails. 100 = no clip.")
    ap.add_argument("--xticks-every", type=int, default=None,
                    help="(Slot) label every Nth position on x-axis.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.dimension == "layer":
        M, a_vec, b_vec, qids, cols, xaxis_word = _load_layer(
            Path(args.input), args.eval_name, args.direction,
            args.baseline_a_label, args.baseline_b_label,
        )
    else:
        M, a_vec, b_vec, qids, cols, xaxis_word = _load_slot(
            Path(args.input), args.baseline_a_label, args.baseline_b_label,
        )

    base_vec, donor_vec, base_name, donor_name = _direction_sides(
        args.direction, a_vec, b_vec, args.baseline_a_label, args.baseline_b_label,
    )

    if args.sort_by == "trained":
        order = np.argsort(a_vec)
    elif args.sort_by == "untrained":
        order = np.argsort(b_vec)
    elif args.sort_by == "gap":
        order = np.argsort(b_vec - a_vec)
    else:
        order = np.arange(len(qids))
    M = M[order]
    a_vec = a_vec[order]
    b_vec = b_vec[order]
    base_vec = base_vec[order]
    donor_vec = donor_vec[order]
    qids = [qids[i] for i in order]

    if args.group_by_type:
        def qtype(qid: str) -> str:
            for t in TYPE_ORDER:
                if qid.startswith(t):
                    return t
            return "other"
        types = np.array([qtype(q) for q in qids])
        rows, a_new, b_new, base_new, donor_new, labels = [], [], [], [], [], []
        for t in TYPE_ORDER:
            mask = types == t
            if not mask.any():
                continue
            rows.append(M[mask].mean(axis=0))
            a_new.append(a_vec[mask].mean())
            b_new.append(b_vec[mask].mean())
            base_new.append(base_vec[mask].mean())
            donor_new.append(donor_vec[mask].mean())
            labels.append(f"{t}  (n={mask.sum()})")
        M = np.stack(rows)
        a_vec = np.array(a_new)
        b_vec = np.array(b_new)
        base_vec = np.array(base_new)
        donor_vec = np.array(donor_new)
        qids = labels

    if args.metric == "delta_base":
        M_plot = M - base_vec[:, None]
        donor_plot = donor_vec - base_vec
        cmap = "RdBu_r"
        abs_vals = np.abs(M_plot)
        if args.clip_percentile < 100:
            vmax = float(np.nanpercentile(abs_vals, args.clip_percentile))
        else:
            vmax = float(np.nanmax(abs_vals))
        vmax = max(vmax, 1e-6)
        vmin = -vmax
        cbar_label = (f"log-perplexity (swap - {base_name})"
                      + (f", clipped p{args.clip_percentile:g}"
                         if args.clip_percentile < 100 else ""))
    else:
        M_plot = M
        donor_plot = donor_vec
        cmap = "viridis_r"
        all_vals = np.concatenate([M_plot.ravel(), donor_vec])
        if args.clip_percentile < 100:
            vmin = float(np.nanpercentile(all_vals, 100 - args.clip_percentile))
            vmax = float(np.nanpercentile(all_vals, args.clip_percentile))
        else:
            vmin = float(np.nanmin(all_vals))
            vmax = float(np.nanmax(all_vals))
        cbar_label = "log-perplexity (per question)"

    n_cols = M_plot.shape[1]
    col_span_ratio = 60 if args.dimension == "slot" else max(1, n_cols * 0.9)
    row_height = 0.55 if args.group_by_type else 0.14
    width_extra = 0.022 * n_cols if args.dimension == "slot" else 0.22 * n_cols
    fig, axes = plt.subplots(
        1, 2, figsize=(3 + width_extra, row_height * len(qids) + 2),
        gridspec_kw={"width_ratios": [1, col_span_ratio]},
        sharey=False,
    )
    ax_base, ax_grid = axes

    # Reference strip: donor (trained) baseline per question.
    ax_base.imshow(donor_plot[:, None], aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax_base.set_xticks([0])
    ax_base.set_xticklabels([donor_name], rotation=45, ha="right", fontsize=8)
    ax_base.set_yticks(range(len(qids)))
    label_fs = 9 if args.group_by_type else 5
    ax_base.set_yticklabels(qids, fontsize=label_fs)
    ax_base.tick_params(axis="y", labelleft=True)

    im = ax_grid.imshow(M_plot, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    if args.dimension == "layer":
        ax_grid.set_xticks(range(n_cols))
        ax_grid.set_xticklabels(cols, rotation=0, fontsize=7)
    else:
        step = args.xticks_every or max(1, n_cols // 20)
        ticks = list(range(0, n_cols, step))
        if ticks[-1] != n_cols - 1:
            ticks.append(n_cols - 1)
        ax_grid.set_xticks(ticks)
        ax_grid.set_xticklabels(ticks, rotation=0, fontsize=7)
    ax_grid.tick_params(axis="y", labelleft=False)

    dir_phrase = (f"{donor_name} -> {base_name}")
    ax_grid.set_xlabel(f"{xaxis_word} ({dir_phrase})")

    if args.group_by_type:
        for r in range(M_plot.shape[0]):
            for c in range(M_plot.shape[1]):
                ax_grid.text(c, r, f"{M_plot[r, c]:.1f}", ha="center",
                             va="center", fontsize=5)

    cbar = fig.colorbar(im, ax=ax_grid, fraction=0.02, pad=0.01)
    cbar.set_label(cbar_label, fontsize=9)

    title = args.title or (
        f"{dir_phrase} per-question log-perplexity "
        f"({args.baseline_a_label}={a_vec.mean():.3f}, "
        f"{args.baseline_b_label}={b_vec.mean():.3f})"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
