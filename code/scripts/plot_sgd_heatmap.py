"""
Plot heatmaps for SGD baseline experiments: Learning Rate vs Batch Size.

This script creates 2D heatmap visualizations showing how SGD performance varies
across the learning rate and batch size grid.

Usage:
    python scripts/plot_sgd_heatmap.py [--epoch EPOCH] [--metric METRIC]
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
)
from optexp.plotter.data_utils import load_data_for_exps


def plot_sgd_heatmap(
    experiments,
    metric_name="va_Accuracy",
    epoch=None,
):
    """
    Create heatmaps for SGD showing learning rate vs batch size.
    Creates separate plots for with and without momentum.
    
    Args:
        experiments: List of experiments
        metric_name: Metric to visualize (default: validation accuracy)
        epoch: Which epoch to plot (default: last epoch)
    """
    if epoch is None:
        epoch = experiments[0].epochs
    
    # Load experiment data
    exps_w_data = load_data_for_exps(experiments)
    
    all_metrics = list(set([metric_name, "tr_Accuracy", "tr_CrossEntropyLoss", "va_CrossEntropyLoss"]))
    
    # Build dataframe from experiment data
    list_of_dicts = []
    for exp_w_data in exps_w_data:
        exp = exp_w_data["exp"]
        df = exp_w_data["data"]
        
        if len(df.index) == 0:
            continue
        
        row = {
            "seed": exp.seed,
            "lr": exp.optim.learning_rate.as_float(),
            "exp_id": exp.exp_id(),
            "group": exp.group,
            "opt": exp.optim.__class__.__name__,
            "batch_size": exp.problem.dataset.batch_size,
        }
        
        # Extract momentum
        if hasattr(exp.optim, 'momentum'):
            row["momentum"] = exp.optim.momentum
        else:
            row["momentum"] = 0.0
        
        # Extract raw metric values at the target epoch
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
    print(f"Unique optimizers: {exps_df['opt'].unique()}")
    
    # Filter for SGD optimizer
    exps_df = exps_df[exps_df["opt"] == "SGD"]
    print(f"\n=== DEBUG: After filtering for SGD ===")
    print(f"Rows remaining: {len(exps_df)}")
    
    if len(exps_df) == 0:
        print("\nERROR: No SGD experiments found!")
        print("Make sure INCLUDE_BASELINES=True in barcoded_mnist_adaptive_sign.py")
        return
    
    print(f"Unique batch_sizes: {sorted(exps_df['batch_size'].unique())}")
    print(f"Unique lrs: {sorted(exps_df['lr'].unique())}")
    print(f"Unique momentum values: {sorted(exps_df['momentum'].unique())}")
    
    # Create figure with 2 subplots (without momentum, with momentum)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (momentum_value, variant_name) in enumerate([(0.0, "without Momentum"), (0.9, "with Momentum")]):
        print(f"\n=== DEBUG: Processing {variant_name} (momentum={momentum_value}) ===")
        
        # Filter by momentum
        momentum_df = exps_df[exps_df["momentum"] == momentum_value]
        
        print(f"Rows for this momentum: {len(momentum_df)}")
        
        if momentum_df.empty:
            print(f"Warning: No data for {variant_name}")
            axes[idx].text(0.5, 0.5, f"No data for\n{variant_name}", 
                          ha="center", va="center", transform=axes[idx].transAxes)
            axes[idx].set_title(f"SGD {variant_name}")
            continue
        
        print(f"{metric_name} range: [{momentum_df[metric_name].min():.4f}, {momentum_df[metric_name].max():.4f}]")
        
        # Create pivot table: rows=learning_rate, columns=batch_size
        pivot_data = momentum_df.pivot_table(
            values=metric_name,
            index="lr",
            columns="batch_size",
            aggfunc="mean"
        )
        
        # Sort: learning rate descending (largest at top), batch size ascending (left to right)
        pivot_data = pivot_data.sort_index(ascending=False)
        pivot_data = pivot_data.sort_index(axis=1, ascending=True)
        
        data = pivot_data.to_numpy(dtype=float, copy=True)
        nrows, ncols = data.shape
        lr_labels = [f"{lr:.0e}" for lr in pivot_data.index]
        batch_labels = [str(int(bs)) for bs in pivot_data.columns]
        
        print(f"\n=== DEBUG: Pivot table for {variant_name} ===")
        print(f"Shape: {data.shape}, dtype: {data.dtype}")
        print(f"Values:\n{data}")
        print(f"Min={np.nanmin(data):.4f}, Max={np.nanmax(data):.4f}")
        
        ax = axes[idx]
        cmap = "YlOrRd_r" if "loss" in metric_name.lower() else "RdYlGn"
        
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=np.nanmin(data), vmax=np.nanmax(data))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name)
        
        # Annotate each cell with its value
        for i in range(nrows):
            for j in range(ncols):
                val = data[i, j]
                if np.isfinite(val):
                    # Pick text color for readability
                    norm_val = im.norm(val)
                    rgba = im.cmap(norm_val)
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "white" if luminance < 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            color=text_color, fontsize=10, weight="bold")
        
        ax.set_title(f"SGD {variant_name}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Learning Rate")
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.set_xticklabels(batch_labels)
        ax.set_yticklabels(lr_labels)
    
    plt.suptitle(
        f"SGD Baseline: {metric_name} at Epoch {epoch}",
        fontsize=14,
        y=0.98
    )
    plt.tight_layout()
    
    # Save figure
    save_dir = config.get_plots_directory() / "AdaptiveSign_ImbalancedMNIST_CNN" / "baselines"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"SGD_{metric_name}_epoch{epoch}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {filepath}")
    
    plt.close(fig)


def plot_all_sgd_metrics(experiments, epoch=None):
    """Plot SGD heatmaps for all key metrics."""
    metrics = [
        "va_Accuracy",
        "tr_Accuracy", 
        "va_CrossEntropyLoss",
        "tr_CrossEntropyLoss",
    ]
    
    for metric in metrics:
        print(f"\n{'='*80}")
        print(f"Plotting {metric} for SGD...")
        print(f"{'='*80}")
        try:
            plot_sgd_heatmap(
                experiments,
                metric_name=metric,
                epoch=epoch,
            )
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SGD baseline heatmaps")
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
        "--group",
        type=str,
        default=None,
        help="WandB group name to filter experiments"
    )
    
    args = parser.parse_args()
    
    # Filter experiments by group if specified
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
        plot_sgd_heatmap(
            exps_to_plot,
            metric_name=args.metric,
            epoch=args.epoch,
        )
    else:
        # Plot all metrics
        plot_all_sgd_metrics(exps_to_plot, epoch=args.epoch)
