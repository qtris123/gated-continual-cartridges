import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    torch = None


def to_numpy(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_selection_matrix(log, n_slots=1024):
    num_steps = len(log)
    mat = np.zeros((num_steps, n_slots), dtype=np.uint8)
    positions_per_step = []
    step_labels = []

    for row_idx, record in enumerate(log):
        pos = to_numpy(record["top_positions"]).astype(int).ravel()

        pos = pos[(pos >= 0) & (pos < n_slots)]
        pos = np.unique(pos)

        mat[row_idx, pos] = 1
        positions_per_step.append(pos)

        # Use the actual recorded step value
        step_labels.append(record.get("step", row_idx))

    return mat, positions_per_step, step_labels


def plot_top_positions(log, n_slots=1024, bin_size=16, save_path=None):
    mat, positions_per_step, step_labels = build_selection_matrix(
        log,
        n_slots=n_slots
    )

    num_steps = len(log)
    n_bins = int(np.ceil(n_slots / bin_size))

    padded_width = n_bins * bin_size
    padded = np.zeros((num_steps, padded_width), dtype=np.uint8)
    padded[:, :n_slots] = mat

    binned = padded.reshape(num_steps, n_bins, bin_size).sum(axis=2)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.4, 1.4])

    # --------------------------------------------------
    # 1. Exact selected slots
    # --------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    for row_idx, pos in enumerate(positions_per_step):
        y = np.full_like(pos, row_idx)
        ax1.scatter(pos, y, marker="|", s=120, linewidths=1.2)

    ax1.set_title("Exact selected top-t slots per recorded step")
    ax1.set_xlabel("Slot index")
    ax1.set_ylabel("Recorded step")
    ax1.set_xlim(-1, n_slots)
    ax1.set_yticks(range(num_steps))
    ax1.set_yticklabels([f"step {s}" for s in step_labels])
    ax1.grid(axis="x", alpha=0.25)

    # --------------------------------------------------
    # 2. Binary heatmap
    # --------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    im2 = ax2.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )

    ax2.set_title("Binary heatmap: selected = 1, not selected = 0")
    ax2.set_xlabel("Slot index")
    ax2.set_ylabel("Recorded step")
    ax2.set_yticks(range(num_steps))
    ax2.set_yticklabels([f"step {s}" for s in step_labels])
    fig.colorbar(im2, ax=ax2, label="selected")

    # --------------------------------------------------
    # 3. Binned density heatmap
    # --------------------------------------------------
    ax3 = fig.add_subplot(gs[2])

    im3 = ax3.imshow(
        binned,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )

    ax3.set_title(f"Binned density heatmap, bin size = {bin_size} slots")
    ax3.set_xlabel("Slot bin start index")
    ax3.set_ylabel("Recorded step")
    ax3.set_yticks(range(num_steps))
    ax3.set_yticklabels([f"step {s}" for s in step_labels])

    tick_bins = np.linspace(0, n_bins - 1, min(9, n_bins), dtype=int)
    ax3.set_xticks(tick_bins)
    ax3.set_xticklabels([f"{b * bin_size}" for b in tick_bins])

    fig.colorbar(im3, ax=ax3, label="# selected slots in bin")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    return mat, binned