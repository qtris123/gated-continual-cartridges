"""
Visualize TF and TF-IDF score distributions from tfidf_ranking_log.pt.

Two plot types:
  1. Single-step heatmap  — TF and TF-IDF side-by-side for one step
  2. Multi-step comparison — pick a fixed layer (or slot) and track how
     scores evolve across logged steps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ────────────────────────────────────────────────────────────────────
PATH = (
    "/home/vo43/cartridges/outputs/"
    "2026-06-13-14-40-21-continual/"
    "32c0b742-e17c-4790-927b-3a8b4a3505ab/"
    "tfidf_ranking_log.pt"
)

# ── Zoom controls (apply to both plot types) ──────────────────────────────────
layer_range = None   # e.g. (0, 8)    → layers 0-7   | None = all
slot_range  = None   # e.g. (0, 256)  → slots 0-255  | None = all

# ── Plot 1 controls ───────────────────────────────────────────────────────────
step_idx = 0   # index into per_step_info list (not the optimizer step number)

# ── Plot 2 controls ───────────────────────────────────────────────────────────
# Choose ONE of the two modes by setting the other to None.
compare_layer = 0    # show score vector (over slots) for this layer, across steps
compare_slot  = None # show score vector (over layers) for this slot, across steps
# Which steps to compare; None = all logged steps (can be slow if many)
step_indices  = None  # e.g. [0, 5, 10, -1]

# ─────────────────────────────────────────────────────────────────────────────

log = torch.load(PATH, map_location="cpu")
steps = log["per_step_info"]
n_steps = len(steps)
n_layers, n_slots = steps[0]["tf"].shape

print(f"Loaded {n_steps} logged steps, shape per step: {n_layers} layers × {n_slots} slots")
print(f"Optimizer steps logged: {[s['step'] for s in steps]}")

# ── helpers ───────────────────────────────────────────────────────────────────

def apply_zoom(tensor):
    """Slice [n_layers, n_slots] tensor according to layer_range / slot_range."""
    ls = layer_range[0] if layer_range else 0
    le = min(layer_range[1], n_layers) if layer_range else n_layers
    ss = slot_range[0] if slot_range else 0
    se = min(slot_range[1], n_slots) if slot_range else n_slots
    return tensor[ls:le, ss:se].float(), ls, le, ss, se


def layer_labels(ls, le):
    return list(range(ls, le))


def slot_tick_fmt(ss):
    return ticker.FuncFormatter(lambda x, _: str(int(x + ss)))


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — single-step heatmap (TF  |  TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════

info = steps[step_idx]
opt_step = info["step"]

tf_view,    ls, le, ss, se = apply_zoom(info["tf"])
tfidf_view, *_             = apply_zoom(info["tfidf"])

layers_shown = layer_labels(ls, le)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(f"TF and TF-IDF scores — optimizer step {opt_step}  "
             f"(log entry {step_idx}/{n_steps-1})", fontsize=12)

for ax, data, title, cmap in zip(
    axes,
    [tf_view, tfidf_view],
    ["TF  (normalized attention)", "TF-IDF  (TF × IDF)"],
    ["Blues", "Greens"],
):
    # Clip negative values (artefacts from IDF with zero-freq slots)
    data_plot = data.clamp(min=0).numpy()
    # Use percentile-based vmax so a few outlier slots don't wash out the rest.
    vmax = float(np.percentile(data_plot[data_plot > 0], 99)) if (data_plot > 0).any() else data_plot.max()
    im = ax.imshow(data_plot, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, extend="max")   # "max" arrow shows values above cap
    ax.set_title(title)
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(layers_shown)))
    ax.set_yticklabels(layers_shown, fontsize=7)
    ax.xaxis.set_major_formatter(slot_tick_fmt(ss))

plt.tight_layout()
plt.savefig(f"plot1_step{step_idx}_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: plot1_step{step_idx}_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — multi-step comparison
# ══════════════════════════════════════════════════════════════════════════════

chosen_indices = step_indices if step_indices is not None else list(range(n_steps))
chosen_indices = [i % n_steps for i in chosen_indices]   # support negative indexing

if compare_layer is not None and compare_slot is None:
    # ── mode A: fixed layer, x = slot, lines = steps ─────────────────────────
    ss = slot_range[0] if slot_range else 0
    se = min(slot_range[1], n_slots) if slot_range else n_slots
    x  = np.arange(ss, se)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Score evolution across steps — Layer {compare_layer}", fontsize=12)

    cmap_steps = plt.cm.plasma(np.linspace(0.1, 0.9, len(chosen_indices)))

    for ax, score_key, ylabel in zip(axes, ["tf", "tfidf"], ["TF", "TF-IDF"]):
        for color, sidx in zip(cmap_steps, chosen_indices):
            s = steps[sidx]
            vec = s[score_key][compare_layer, ss:se].float().clamp(min=0).numpy()
            ax.plot(x, vec, color=color, linewidth=0.8, alpha=0.8,
                    label=f"step {s['step']}")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at layer {compare_layer}")

    axes[-1].set_xlabel("Slot index")
    handles, labels = axes[0].get_legend_handles_labels()
    # show at most 10 legend entries to keep it readable
    step = max(1, len(handles) // 10)
    axes[0].legend(handles[::step], labels[::step], fontsize=7,
                   loc="upper right", ncol=2)

    plt.tight_layout()
    fname = f"plot2_layer{compare_layer}_multistep.png"

elif compare_slot is not None and compare_layer is None:
    # ── mode B: fixed slot, x = layer, lines = steps ─────────────────────────
    ls = layer_range[0] if layer_range else 0
    le = min(layer_range[1], n_layers) if layer_range else n_layers
    x  = np.arange(ls, le)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Score evolution across steps — Slot {compare_slot}", fontsize=12)

    cmap_steps = plt.cm.plasma(np.linspace(0.1, 0.9, len(chosen_indices)))

    for ax, score_key, ylabel in zip(axes, ["tf", "tfidf"], ["TF", "TF-IDF"]):
        for color, sidx in zip(cmap_steps, chosen_indices):
            s = steps[sidx]
            vec = s[score_key][ls:le, compare_slot].float().clamp(min=0).numpy()
            ax.plot(x, vec, color=color, linewidth=1.2, alpha=0.8, marker="o",
                    markersize=3, label=f"step {s['step']}")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at slot {compare_slot}")

    axes[-1].set_xlabel("Layer")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    step = max(1, len(handles) // 10)
    axes[0].legend(handles[::step], labels[::step], fontsize=7,
                   loc="upper right", ncol=2)

    plt.tight_layout()
    fname = f"plot2_slot{compare_slot}_multistep.png"

else:
    raise ValueError("Set exactly one of compare_layer or compare_slot (not both, not neither).")

plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {fname}")
