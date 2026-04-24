"""Per-question swap heatmaps for *_per_question experiment folders (explore.ipynb layout)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PT_NAME = "per_slot_swap.pt"


def parse_folder(name: str) -> tuple[str, str, str]:
    m = re.match(r"per_token_slot_([\w]+)_toks(\d+)_(.+)$", name)
    if not m:
        return name, "", ""
    return m.group(1), m.group(2), m.group(3)


def plot_one(exp_dir: Path, data: dict, model: str, toks: str) -> None:
    grid_per_q = data["grid_per_q"].detach().cpu()
    base_per_q = data["base_per_q"].detach().cpu()
    donor_per_q = data["donor_per_q"].detach().cpu()
    original_q_ids = list(data["question_ids"])

    num_questions, num_slots = grid_per_q.shape
    sorted_indices = sorted(range(len(original_q_ids)), key=lambda k: original_q_ids[k])
    sorted_q_ids = [original_q_ids[i] for i in sorted_indices]
    idx_tensor = torch.tensor(sorted_indices)

    sorted_grid = grid_per_q[idx_tensor]
    sorted_base = base_per_q[idx_tensor]
    sorted_donor = donor_per_q[idx_tensor]

    main_grid_data = sorted_grid - sorted_base.unsqueeze(1)
    left_col_data = (sorted_donor - sorted_base).unsqueeze(1)

    vmin, vmax = -1.0, 1.0
    clipped_grid =  main_grid_data.numpy() #torch.clamp(main_grid_data, vmin, vmax).numpy()
    clipped_col = left_col_data.numpy() #torch.clamp(left_col_data, vmin, vmax).numpy()

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 25, 0.7], wspace=0.015)
    cmap_val = "RdBu_r"

    ax_col = fig.add_subplot(gs[0])
    ax_col.imshow(clipped_col, aspect="auto", cmap=cmap_val, vmin=vmin, vmax=vmax)
    ax_col.set_yticks(np.arange(num_questions))
    ax_col.set_yticklabels(sorted_q_ids, fontsize=6)
    ax_col.set_ylabel("Questions", fontsize=12)
    ax_col.set_xticks([])
    ax_col.set_xlabel("donor - base", fontsize=9)

    ax_grid = fig.add_subplot(gs[1])
    im_grid = ax_grid.imshow(clipped_grid, aspect="auto", cmap=cmap_val) #, vmin=vmin, vmax=vmax)
    ax_grid.set_yticks([])
    ax_grid.set_xlabel("slot position p", fontsize=14)
    title = (
        f"{model} toks={toks} | trained -> untrained per question: "
        f"log-perplexity (swap - untrained) toks={num_slots}"
    )
    ax_grid.set_title(title, fontsize=18, pad=15)

    cbar_ax = fig.add_subplot(gs[2])
    cbar = fig.colorbar(im_grid, cax=cbar_ax, orientation="vertical")
    cbar.set_label(
        "log-perplexity difference, clipped at [-1.0, 1.0]",
        rotation=270,
        labelpad=25,
        fontsize=11,
    )

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.08, right=0.95)
    out = exp_dir / "per_question_heatmap_no_clip.png"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    candidates = sorted(
        p
        for p in ROOT.iterdir()
        if p.is_dir()
        and p.name.endswith("_per_question")
        and (p / PT_NAME).is_file()
    )
    if not candidates:
        raise SystemExit(f"No *_per_question folders with {PT_NAME} under {ROOT}")

    for exp_dir in candidates:
        data = torch.load(exp_dir / PT_NAME, map_location="cpu", weights_only=False)
        if "grid_per_q" not in data:
            print(f"Skip (no grid_per_q): {exp_dir.name}")
            continue
        model, toks, _ = parse_folder(exp_dir.name)
        plot_one(exp_dir, data, model, toks)


if __name__ == "__main__":
    main()
