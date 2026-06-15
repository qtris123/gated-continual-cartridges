#!/usr/bin/env python3
"""
Generate visualization plots from training output folder.

Usage:
    python generate_plots.py /path/to/output/folder
    
    # Or with specific options:
    python generate_plots.py /path/to/output/folder --n_slots 1024 --bin_size 16 --top_k 50

This script generates:
    1. top_positions_heatmap.png - from sparse_slot_log.pt
    2. slot_importance_heatmap.png - from bg_stats.pt
"""

import argparse
import os
import sys

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

# Add the viz directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_top_positions import plot_top_positions
from plot_slot_importance_heatmaps import plot_slot_importance_heatmaps


def generate_plots(output_folder: str, n_slots: int = 1024, bin_size: int = 16, top_k: int = 50):
    """
    Generate visualization plots from training output folder.
    
    Args:
        output_folder: Path to the training output folder containing .pt files
        n_slots: Number of cache slots (default: 1024)
        bin_size: Bin size for top positions heatmap (default: 16)
        top_k: Top-k threshold for slot importance analysis (default: 50)
    """
    print(f"Generating plots for: {output_folder}")
    
    # Check if folder exists
    if not os.path.isdir(output_folder):
        print(f"Error: Output folder not found: {output_folder}")
        return False
    
    plots_generated = 0
    
    # 1. Generate top positions heatmap from sparse_slot_log.pt
    sparse_log_path = os.path.join(output_folder, "sparse_slot_log.pt")
    if os.path.exists(sparse_log_path):
        print(f"Loading sparse_slot_log.pt...")
        log = torch.load(sparse_log_path, weights_only=False)
        
        if log:
            save_path = os.path.join(output_folder, "top_positions_heatmap.png")
            print(f"Generating top positions heatmap ({len(log)} steps)...")
            
            # Temporarily disable interactive display
            plt.ioff()
            mat, binned = plot_top_positions(
                log,
                n_slots=n_slots,
                bin_size=bin_size,
                save_path=save_path
            )
            plt.close('all')
            
            print(f"Saved: {save_path}")
            plots_generated += 1
        else:
            print("Warning: sparse_slot_log.pt is empty")
    else:
        print(f"Note: sparse_slot_log.pt not found, skipping top positions plot")
    
    # 2. Generate slot importance heatmap from bg_stats.pt
    bg_stats_path = os.path.join(output_folder, "bg_stats.pt")
    if os.path.exists(bg_stats_path):
        print(f"Loading bg_stats.pt...")
        bg = torch.load(bg_stats_path, weights_only=False)
        
        if "batch_ranked_positions" in bg:
            save_path = os.path.join(output_folder, "slot_importance_heatmap.png")
            print(f"Generating slot importance heatmap ({bg['num_batches']} batches)...")
            
            plt.ioff()
            results = plot_slot_importance_heatmaps(
                bg["batch_ranked_positions"],
                n_slots=n_slots,
                top_k=top_k,
                reorder="cluster",
                save_path=save_path
            )
            plt.close('all')
            
            print(f"Saved: {save_path}")
            plots_generated += 1
        else:
            print("Warning: bg_stats.pt does not contain batch_ranked_positions")
    else:
        print(f"Note: bg_stats.pt not found, skipping slot importance plot")
    
    # 3. Generate TF-IDF ranking log visualization if available
    tfidf_log_path = os.path.join(output_folder, "tfidf_ranking_log.pt")
    if os.path.exists(tfidf_log_path):
        print(f"Loading tfidf_ranking_log.pt...")
        tfidf_data = torch.load(tfidf_log_path, weights_only=False)
        
        if tfidf_data.get("per_step_info"):
            save_path = os.path.join(output_folder, "tfidf_scores_heatmap.png")
            print(f"Generating TF-IDF scores heatmap...")
            
            plt.ioff()
            _generate_tfidf_heatmap(tfidf_data, n_slots, save_path)
            plt.close('all')
            
            print(f"Saved: {save_path}")
            plots_generated += 1
    
    print(f"\nDone! Generated {plots_generated} plot(s)")
    return plots_generated > 0


def _generate_tfidf_heatmap(tfidf_data, n_slots, save_path):
    """Generate a heatmap showing TF and TF-IDF values over steps."""
    import numpy as np
    
    per_step = tfidf_data["per_step_info"]
    num_steps = len(per_step)
    
    if num_steps == 0:
        return
    
    # Build matrices
    tf_matrix = np.zeros((num_steps, n_slots))
    tfidf_matrix = np.zeros((num_steps, n_slots))
    steps = []
    
    for i, info in enumerate(per_step):
        tf = info["tf"].numpy() if hasattr(info["tf"], "numpy") else info["tf"]
        tfidf = info["tfidf"].numpy() if hasattr(info["tfidf"], "numpy") else info["tfidf"]
        tf_matrix[i, :len(tf)] = tf
        tfidf_matrix[i, :len(tfidf)] = tfidf
        steps.append(info["step"])
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # TF heatmap
    im1 = axes[0].imshow(tf_matrix, aspect="auto", interpolation="nearest", origin="lower")
    axes[0].set_title("TF (Term Frequency) per step")
    axes[0].set_xlabel("Slot ID")
    axes[0].set_ylabel("Step")
    axes[0].set_yticks(range(num_steps))
    axes[0].set_yticklabels([f"step {s}" for s in steps])
    fig.colorbar(im1, ax=axes[0], label="TF value")
    
    # TF-IDF heatmap
    im2 = axes[1].imshow(tfidf_matrix, aspect="auto", interpolation="nearest", origin="lower")
    axes[1].set_title("TF-IDF per step")
    axes[1].set_xlabel("Slot ID")
    axes[1].set_ylabel("Step")
    axes[1].set_yticks(range(num_steps))
    axes[1].set_yticklabels([f"step {s}" for s in steps])
    fig.colorbar(im2, ax=axes[1], label="TF-IDF value")
    
    # Add IDF info if available
    if tfidf_data.get("idf") is not None:
        idf = tfidf_data["idf"]
        if hasattr(idf, "numpy"):
            idf = idf.numpy()
        fig.suptitle(f"TF-IDF Analysis (IDF range: {idf.min():.3f} - {idf.max():.3f})", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from training output folder"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the training output folder"
    )
    parser.add_argument(
        "--n_slots",
        type=int,
        default=1024,
        help="Number of cache slots (default: 1024)"
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=16,
        help="Bin size for top positions heatmap (default: 16)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k threshold for slot importance analysis (default: 50)"
    )
    
    args = parser.parse_args()
    
    success = generate_plots(
        output_folder=args.output_folder,
        n_slots=args.n_slots,
        bin_size=args.bin_size,
        top_k=args.top_k
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
