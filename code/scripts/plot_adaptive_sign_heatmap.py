"""
Plot heatmaps for AdaptiveSign experiments: Learning Rate vs Epsilon.

This script creates 2D heatmap visualizations showing how performance varies
across the learning rate and epsilon hyperparameter grid.

Experiment grid:
- Learning rates: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
- Epsilon values: [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
- Batch sizes: [64, 256, 1024]
- Momentum variants: with (M) and without (NM)
- Epochs: 20

Usage:
    python scripts/plot_adaptive_sign_heatmap.py [--epoch EPOCH] [--metric METRIC]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optexp import config
from optexp.experiments.vision.barcoded_mnist_adaptive_sign import (
    experiments,
    BATCH_SIZES,
    EPSILON_VALUES,
)
from optexp.plotter.data_utils import load_data_for_exps


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
        batch_sizes = BATCH_SIZES
    
    # Load experiment data
    exps_w_data = load_data_for_exps(experiments)
    
    all_metrics = list(set([metric_name, "tr_Accuracy", "tr_CrossEntropyLoss", "va_CrossEntropyLoss"]))
    
    # Build dataframe directly from experiment data to avoid thresholding/clamping
    # in get_exps_data_epoch which replaces diverged values with initial values
    list_of_dicts = []
    for exp_w_data in exps_w_data:
        exp = exp_w_data["exp"]
        df = exp_w_data["data"]
        
        if len(df.index) == 0:
            continue
        
        # Filter for AdaptiveSign only (SGD/Adam don't have eps parameter)
        if exp.optim.__class__.__name__ != "AdaptiveSign":
            continue
        
        row = {
            "seed": exp.seed,
            "lr": exp.optim.learning_rate.as_float(),
            "exp_id": exp.exp_id(),
            "group": exp.group,
            "opt": exp.optim.__class__.__name__,
            "batch_size": exp.problem.dataset.batch_size,
            "eps": exp.optim.eps,
        }
        
        # Extract momentum
        if hasattr(exp.optim, 'beta1'):
            row["momentum"] = exp.optim.beta1
        elif hasattr(exp.optim, 'momentum'):
            row["momentum"] = exp.optim.momentum
        else:
            row["momentum"] = 0.0
        
        # Extract raw metric values at the target epoch (no clamping)
        df_indexed = df.set_index("epoch")
        for m in all_metrics:
            if m not in df_indexed.columns:
                row[m] = np.nan
                continue
            if epoch in df_indexed.index:
                val = df_indexed.loc[epoch, m]
                row[m] = val if np.isfinite(val) else np.nan
            else:
                # Epoch not reached — use last available value
                last_valid = df_indexed[m].last_valid_index()
                if last_valid is not None:
                    row[m] = df_indexed.loc[last_valid, m]
                else:
                    row[m] = np.nan
        
        list_of_dicts.append(row)
    
    exps_df = pd.DataFrame(list_of_dicts)
    
    print(f"\n=== DEBUG: Built DataFrame ===")
    print(f"Shape: {exps_df.shape}")
    print(f"Columns: {exps_df.columns.tolist()}")
    print(f"Unique batch_sizes: {sorted(exps_df['batch_size'].dropna().unique())}")
    print(f"Unique eps values: {sorted(exps_df['eps'].dropna().unique())}")
    print(f"Unique lrs: {sorted(exps_df['lr'].dropna().unique())}")
    print(f"Sample rows:\n{exps_df[['exp_id', 'batch_size', 'eps', 'lr', 'opt', 'momentum', metric_name]].head(10)}")
    
    # Filter for AdaptiveSign optimizer
    exps_df = exps_df[exps_df["opt"] == "AdaptiveSign"]
    print(f"\n=== DEBUG: After filtering for AdaptiveSign ===")
    print(f"Rows remaining: {len(exps_df)}")
    
    # Filter by momentum variant
    if momentum_variant == "M":
        exps_df = exps_df[exps_df["momentum"] > 0]
        variant_name = "with Momentum"
    else:
        exps_df = exps_df[exps_df["momentum"] == 0]
        variant_name = "without Momentum"
    
    print(f"\n=== DEBUG: After filtering for {variant_name} ===")
    print(f"Rows remaining: {len(exps_df)}")
    print(f"Momentum values: {exps_df['momentum'].unique()}")
    
    # Create figure with subplots for each batch size
    n_batches = len(batch_sizes)
    fig, axes = plt.subplots(1, n_batches, figsize=(6*n_batches, 5))
    if n_batches == 1:
        axes = [axes]
    
    for idx, batch_size in enumerate(batch_sizes):
        print(f"\n=== DEBUG: Processing batch_size={batch_size} ===")
        
        # Filter by batch size
        batch_df = exps_df[exps_df["batch_size"] == batch_size]
        
        print(f"Rows for this batch_size: {len(batch_df)}")
        
        if batch_df.empty:
            print(f"Warning: No data for batch_size={batch_size}")
            continue
        
        print(f"Unique LRs: {sorted(batch_df['lr'].unique())}")
        print(f"Unique eps: {sorted(batch_df['eps'].unique())}")
        print(f"{metric_name} range: [{batch_df[metric_name].min():.4f}, {batch_df[metric_name].max():.4f}]")
        
        # Create pivot table for heatmap
        # Group by LR and epsilon, take mean across seeds
        pivot_data = batch_df.pivot_table(
            values=metric_name,
            index="eps",
            columns="lr",
            aggfunc="mean"
        )
        
        # Sort: epsilon descending (largest at top), learning rate ascending (left to right)
        pivot_data = pivot_data.sort_index(ascending=False)
        pivot_data = pivot_data.sort_index(axis=1, ascending=True)
        
        # Force to float numpy array — this is critical for imshow color mapping
        data = pivot_data.to_numpy(dtype=float, copy=True)
        
        nrows, ncols = data.shape
        eps_labels = [f"{e:.0e}" for e in pivot_data.index]
        lr_labels = [f"{lr:.0e}" for lr in pivot_data.columns]
        
        print(f"\n=== DEBUG: Pivot table for batch_size={batch_size} ===")
        print(f"Shape: {data.shape}, dtype: {data.dtype}")
        print(f"Values:\n{data}")
        print(f"Min={np.nanmin(data):.4f}, Max={np.nanmax(data):.4f}")
        
        ax = axes[idx]
        cmap = "YlOrRd_r" if "loss" in metric_name.lower() else "RdYlGn"
        
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name)
        
        # Annotate each cell with its value
        for i in range(nrows):
            for j in range(ncols):
                val = data[i, j]
                if np.isfinite(val):
                    # Pick text color for readability against background
                    norm_val = im.norm(val)
                    rgba = im.cmap(norm_val)
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "white" if luminance < 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            color=text_color, fontsize=8, weight="bold")
        
        ax.set_title(f"Batch Size {batch_size} ({variant_name})")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Epsilon")
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.set_xticklabels(lr_labels, rotation=45, ha="right")
        ax.set_yticklabels(eps_labels)
    
    plt.suptitle(
        f"AdaptiveSign {variant_name}: {metric_name} at Epoch {epoch}",
        fontsize=14,
        y=1.02
    )
    plt.tight_layout()
    
    # Save figure
    save_dir = config.get_plots_directory() / "AdaptiveSign_ImbalancedMNIST_CNN" / "heatmaps"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{metric_name}_epoch{epoch}_{momentum_variant}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"Saved: {filepath}")
    
    plt.close(fig)


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
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="WandB group name to plot (default: uses experiments from barcoded_mnist_adaptive_sign.py). Use 'AdaptiveSign_ImbalancedMNIST_CNN' for full-batch results."
    )
    
    args = parser.parse_args()
    
    # If a different group is specified, filter experiments by that group
    exps_to_plot = experiments
    if args.group:
        print(f"Filtering experiments for group: {args.group}")
        exps_to_plot = [exp for exp in experiments if exp.group == args.group]
        if not exps_to_plot:
            print(f"Warning: No experiments found with group '{args.group}'")
            print(f"Available groups: {set(exp.group for exp in experiments)}")
            exit(1)
        print(f"Found {len(exps_to_plot)} experiments in group '{args.group}'")
    
    if args.metric:
        # Plot specific metric
        momentum_variants = ["M", "NM"] if args.momentum == "both" else [args.momentum]
        for variant in momentum_variants:
            plot_heatmap_grid(
                exps_to_plot,
                metric_name=args.metric,
                epoch=args.epoch,
                momentum_variant=variant
            )
    else:
        # Plot all metrics
        plot_all_metrics(exps_to_plot, epoch=args.epoch)


