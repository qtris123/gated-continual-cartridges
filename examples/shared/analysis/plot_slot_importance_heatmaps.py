import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    torch = None

# Optional: for better sample ordering
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def to_numpy(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_rank_matrix(batch_ranked_positions, n_slots=None):
    """
    batch_ranked_positions:
        shape [n_samples, n_ranked_slots]
        Each row contains slot IDs ordered from most important to least important.

    Returns:
        rank_matrix: shape [n_samples, n_slots]
            rank_matrix[i, slot] = rank position of that slot for sample i
            smaller = more important
    """
    ranked = to_numpy(batch_ranked_positions).astype(int)
    n_samples, n_ranked_slots = ranked.shape

    if n_slots is None:
        n_slots = int(ranked.max()) + 1

    # Default fill = n_ranked_slots (or n_slots) for unseen slots
    rank_matrix = np.full((n_samples, n_slots), fill_value=n_ranked_slots, dtype=np.int32)

    row_idx = np.arange(n_samples)[:, None]
    rank_idx = np.arange(n_ranked_slots)[None, :]
    rank_matrix[row_idx, ranked] = rank_idx

    return rank_matrix


def reorder_samples(rank_matrix, method="cluster"):
    """
    Reorder rows so similar samples are near each other.
    """
    n_samples = rank_matrix.shape[0]

    if method == "none":
        return np.arange(n_samples)

    if method == "cluster" and SCIPY_AVAILABLE:
        # Correlation distance on rows
        dist = pdist(rank_matrix, metric="correlation")
        Z = linkage(dist, method="average")
        order = leaves_list(Z)
        return order

    # Fallback: sort by average rank of top slots
    # (not as strong as clustering, but still useful)
    score = rank_matrix[:, :].mean(axis=1)
    return np.argsort(score)


def plot_slot_importance_heatmaps(batch_ranked_positions, n_slots=None, top_k=32, reorder="cluster", save_path=None):
    ranked = to_numpy(batch_ranked_positions).astype(int)
    rank_matrix = build_rank_matrix(ranked, n_slots=n_slots)

    n_samples, n_slots = rank_matrix.shape

    # Reorder rows for visual similarity
    sample_order = reorder_samples(rank_matrix, method=reorder)
    rank_matrix_ord = rank_matrix[sample_order]

    # Normalize ranks to [0, 1] for easier plotting
    rank_norm = rank_matrix_ord / max(1, rank_matrix_ord.shape[1] - 1)

    # Top-k frequency per slot
    topk_mask = rank_matrix < top_k
    topk_freq = topk_mask.mean(axis=0)  # fraction of samples where slot appears in top-k

    # Mean rank per slot
    mean_rank = rank_matrix.mean(axis=0)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[5, 1.5, 1.5])

    # --------------------------------------------------
    # 1) Main rank heatmap
    # --------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(
        rank_norm,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    ax1.set_title("Sample × Slot Rank Heatmap (darker/lower = more important)")
    ax1.set_xlabel("Slot ID")
    ax1.set_ylabel("Sample (reordered by similarity)")
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Normalized rank (0 = most important)")

    # --------------------------------------------------
    # 2) Top-k frequency across samples
    # --------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(np.arange(n_slots), topk_freq)
    ax2.set_title(f"Fraction of samples where slot is in Top-{top_k}")
    ax2.set_xlabel("Slot ID")
    ax2.set_ylabel("Frequency")

    # --------------------------------------------------
    # 3) Mean rank of each slot
    # --------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(np.arange(n_slots), mean_rank)
    ax3.set_title("Mean rank per slot across samples")
    ax3.set_xlabel("Slot ID")
    ax3.set_ylabel("Mean rank (smaller = more important)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    return {
        "rank_matrix": rank_matrix,
        "sample_order": sample_order,
        "topk_frequency": topk_freq,
        "mean_rank": mean_rank,
    }