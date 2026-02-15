"""
Plot heatmaps for AdaptiveSign experiments: Learning Rate vs Epsilon.

This script creates 2D heatmap visualizations showing how performance varies
across the learning rate and epsilon hyperparameter grid.

Usage:
    python scripts/plot_adaptive_sign_heatmap.py [--epoch EPOCH] [--metric METRIC]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optexp import config
from optexp.experiments.vision.barcoded_mnist_adaptive_sign import experiments
from optexp.plotter.data_utils import get_exps_data_epoch, load_data_for_exps


def plot_heatmap_grid(
    experiments,
    metric_name="va_Accuracy",
    epoch=None,
    momentum_variant="M",
    batch_sizes=None,
):
    """
    Create heatmap grids for learning rate vs epsilon.
    
    Args:
        experiments: List of experiments
        metric_name: Metric to visualize (default: validation accuracy)
        epoch: Which epoch to plot (default: last epoch)
        momentum_variant: "M" for with momentum, "NM" for without
        batch_sizes: List of batch sizes to plot (default: all)
    """
    if epoch is None:
        epoch = experiments[0].epochs
    
    if batch_sizes is None:
        batch_sizes = [64, 256, 1024]
    
    # Load experiment data
    exps_w_data = load_data_for_exps(experiments)
    exps_df = get_exps_data_epoch(
        exps_w_data, 
        [metric_name, "tr_Accuracy", "tr_CrossEntropyLoss", "va_CrossEntropyLoss"],
        epoch,
        using_step=False
    )
    
    # Map exp_id to batch_size and epsilon
    exp_metadata = {}
    for exp in experiments:
        exp_id = exp.exp_id()
        exp_metadata[exp_id] = {
            'batch_size': exp.problem.dataset.batch_size,
            'eps': exp.optim.eps
        }
    
    # Add batch_size and epsilon to dataframe using exp_id mapping
    exps_df["batch_size"] = exps_df["exp_id"].map(lambda eid: exp_metadata.get(eid, {}).get('batch_size'))
    exps_df["eps"] = exps_df["exp_id"].map(lambda eid: exp_metadata.get(eid, {}).get('eps'))
    
    # Filter for AdaptiveSign optimizer
    exps_df = exps_df[exps_df["opt"] == "AdaptiveSign"]
    
    # Filter by momentum variant
    if momentum_variant == "M":
        exps_df = exps_df[exps_df["momentum"] > 0]
        variant_name = "with Momentum"
    else:
        exps_df = exps_df[exps_df["momentum"] == 0]
        variant_name = "without Momentum"
    
    # Create figure with subplots for each batch size
    n_batches = len(batch_sizes)
    fig, axes = plt.subplots(1, n_batches, figsize=(6*n_batches, 5))
    if n_batches == 1:
        axes = [axes]
    
    for idx, batch_size in enumerate(batch_sizes):
        # Filter by batch size
        batch_df = exps_df[exps_df["batch_size"] == batch_size]
        
        if batch_df.empty:
            print(f"Warning: No data for batch_size={batch_size}")
            continue
        
        # Create pivot table for heatmap
        # Group by LR and epsilon, take mean across seeds
        pivot_data = batch_df.pivot_table(
            values=metric_name,
            index="eps",
            columns="lr",
            aggfunc="mean"
        )
        
        # Sort: epsilon descending (largest at top), learning rate ascending (left to right)
        pivot_data = pivot_data.sort_index(ascending=False)  # eps descending
        pivot_data = pivot_data.sort_index(axis=1, ascending=True)  # lr ascending
        
        # Create heatmap using matplotlib
        ax = axes[idx]
        
        # Determine colormap and normalization
        if "loss" in metric_name.lower():
            cmap = "YlOrRd"  # Yellow-Orange-Red for loss (lower is better)
            vmin = None
            vmax = None
        else:
            cmap = "RdYlGn"  # Red-Yellow-Green for accuracy (higher is better)
            vmin = 0
            vmax = 1
        
        # Create the heatmap with origin at upper-left
        im = ax.imshow(
            pivot_data.values,
            cmap=cmap,
            aspect="auto",
            interpolation='nearest',
            origin='upper',  # Important: matches sorted data
            vmin=vmin,
            vmax=vmax,
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name)
        
        # Add annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                val = pivot_data.iloc[i, j]
                if not np.isnan(val):
                    # Determine text color based on normalized value for better visibility
                    if vmin is not None and vmax is not None:
                        normalized = (val - vmin) / (vmax - vmin)
                        text_color = "white" if normalized < 0.5 else "black"
                    else:
                        # For loss, use simpler logic
                        text_color = "white"
                    
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center",
                        color=text_color,
                        fontsize=8,
                        weight='bold'
                    )
        
        ax.set_title(f"Batch Size {batch_size} ({variant_name})")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Epsilon")
        
        # Set ticks and labels
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in pivot_data.columns], rotation=45, ha='right')
        ax.set_yticklabels([f"{eps:.0e}" for eps in pivot_data.index], rotation=0)
    
    plt.suptitle(
        f"AdaptiveSign {variant_name}: {metric_name} at Epoch {epoch}",
        fontsize=14,
        y=1.02
    )
    plt.tight_layout()
    
    # Save figure
    save_dir = config.get_plots_directory() / "AdaptiveSign_ImbalancedMNIST_CNN" / "heatmaps"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{metric_name}_epoch{epoch}_{momentum_variant}.pdf"
    filepath = save_dir / filename
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Saved: {filepath}")
    
    plt.show()


def plot_all_metrics(experiments, epoch=None):
    """Plot heatmaps for all key metrics."""
    metrics = [
        "va_Accuracy",
        "tr_Accuracy", 
        "va_CrossEntropyLoss",
        "tr_CrossEntropyLoss",
    ]
    
    for momentum_variant in ["NM", "M"]:
        for metric in metrics:
            print(f"\nPlotting {metric} ({momentum_variant})...")
            try:
                plot_heatmap_grid(
                    experiments,
                    metric_name=metric,
                    epoch=epoch,
                    momentum_variant=momentum_variant
                )
            except Exception as e:
                print(f"Error plotting {metric}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AdaptiveSign heatmaps")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to plot (default: last epoch)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Specific metric to plot (default: plot all)"
    )
    parser.add_argument(
        "--momentum",
        type=str,
        choices=["M", "NM", "both"],
        default="both",
        help="Momentum variant to plot (default: both)"
    )
    
    args = parser.parse_args()
    
    if args.metric:
        # Plot specific metric
        momentum_variants = ["M", "NM"] if args.momentum == "both" else [args.momentum]
        for variant in momentum_variants:
            plot_heatmap_grid(
                experiments,
                metric_name=args.metric,
                epoch=args.epoch,
                momentum_variant=variant
            )
    else:
        # Plot all metrics
        plot_all_metrics(experiments, epoch=args.epoch)
