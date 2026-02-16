"""
Plot majority vs minority class performance for AdaptiveSign experiments.

This script creates visualizations comparing how well the optimizer performs
on majority classes (10 common MNIST classes) vs minority classes (10,230 rare
barcoded classes).

Usage:
    python scripts/plot_majority_minority_stats.py [--epoch EPOCH] [--group GROUP]
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


def extract_majority_minority_metrics(exps_w_data, epoch=None):
    """
    Extract majority and minority metrics from experiment data.
    
    Returns a DataFrame with columns:
    - exp_id, seed, lr, eps, batch_size, momentum, opt
    - va_Accuracy_majority, va_Accuracy_minority
    - tr_Accuracy_majority, tr_Accuracy_minority
    - va_CrossEntropyLoss_majority, va_CrossEntropyLoss_minority
    - tr_CrossEntropyLoss_majority, tr_CrossEntropyLoss_minority
    """
    if epoch is None:
        epoch = exps_w_data[0]["exp"].epochs
    
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
            "eps": exp.optim.eps,
        }
        
        # Extract momentum
        if hasattr(exp.optim, 'beta1'):
            row["momentum"] = exp.optim.beta1
        elif hasattr(exp.optim, 'momentum'):
            row["momentum"] = exp.optim.momentum
        else:
            row["momentum"] = 0.0
        
        # Extract majority/minority metrics at the target epoch
        df_indexed = df.set_index("epoch")
        
        # List of grouped metrics (these return [majority_value, minority_value])
        grouped_metrics = [
            "va_AccuracyMajorityMinority",
            "tr_AccuracyMajorityMinority",
            "va_CrossEntropyLossMajorityMinority",
            "tr_CrossEntropyLossMajorityMinority",
        ]
        
        for metric in grouped_metrics:
            base_name = metric.replace("MajorityMinority", "")
            
            if metric not in df_indexed.columns:
                # Metric not present - check if data was downloaded
                print(f"WARNING: {metric} not found in data for exp {exp.exp_id()}")
                print(f"  Available columns: {df_indexed.columns.tolist()}")
                row[f"{base_name}_majority"] = np.nan
                row[f"{base_name}_minority"] = np.nan
                continue
            
            if epoch in df_indexed.index:
                val = df_indexed.loc[epoch, metric]
            else:
                # Epoch not reached — use last available value
                last_valid = df_indexed[metric].last_valid_index()
                if last_valid is not None:
                    val = df_indexed.loc[last_valid, metric]
                else:
                    val = None
            
            # Parse the list [majority, minority]
            # The value might be stored as a string representation of a list
            if val is not None:
                # Try to parse if it's a string
                if isinstance(val, str):
                    try:
                        import ast
                        val = ast.literal_eval(val)
                    except:
                        print(f"WARNING: Could not parse string value for {metric}: {val}")
                        val = None
                
                if isinstance(val, (list, np.ndarray)) and len(val) == 2:
                    row[f"{base_name}_majority"] = float(val[0])
                    row[f"{base_name}_minority"] = float(val[1])
                else:
                    print(f"WARNING: Unexpected format for {metric}: {type(val)} = {val}")
                    row[f"{base_name}_majority"] = np.nan
                    row[f"{base_name}_minority"] = np.nan
            else:
                # Fallback to NaN
                row[f"{base_name}_majority"] = np.nan
                row[f"{base_name}_minority"] = np.nan
        
        list_of_dicts.append(row)
    
    return pd.DataFrame(list_of_dicts)


def plot_majority_minority_heatmap(
    experiments,
    metric_base="va_Accuracy",
    epoch=None,
    momentum_variant="M",
    batch_sizes=None,
):
    """
    Create side-by-side heatmaps comparing majority vs minority performance.
    
    Args:
        experiments: List of experiments
        metric_base: Base metric name (e.g., "va_Accuracy")
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
    
    # Check if any data was loaded
    data_loaded = sum(1 for exp_w_data in exps_w_data if len(exp_w_data["data"]) > 0)
    print(f"\n=== Data Loading Summary ===")
    print(f"Total experiments: {len(experiments)}")
    print(f"Experiments with data: {data_loaded}")
    if data_loaded == 0:
        print("\nERROR: No experiment data found!")
        print("Have you downloaded the data from WandB?")
        print("Run: python -m optexp.experiments.vision.barcoded_mnist_adaptive_sign --download")
        return
    
    # Extract majority/minority metrics
    exps_df = extract_majority_minority_metrics(exps_w_data, epoch=epoch)
    
    print(f"\n=== DEBUG: Extracted DataFrame ===")
    print(f"Shape: {exps_df.shape}")
    print(f"Columns: {exps_df.columns.tolist()}")
    print(f"Sample rows:\n{exps_df[['exp_id', 'batch_size', 'eps', 'lr', 'opt', 'momentum', f'{metric_base}_majority', f'{metric_base}_minority']].head(10)}")
    
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
    
    # Create figure with subplots: rows=majority/minority, cols=batch_sizes
    n_batches = len(batch_sizes)
    fig, axes = plt.subplots(2, n_batches, figsize=(6*n_batches, 10))
    
    # Ensure axes is always 2D
    if n_batches == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, batch_size in enumerate(batch_sizes):
        print(f"\n=== DEBUG: Processing batch_size={batch_size} ===")
        
        # Filter by batch size
        batch_df = exps_df[exps_df["batch_size"] == batch_size]
        
        print(f"Rows for this batch_size: {len(batch_df)}")
        
        if batch_df.empty:
            print(f"Warning: No data for batch_size={batch_size}")
            continue
        
        # Plot majority (row 0) and minority (row 1)
        for row_idx, class_type in enumerate(["majority", "minority"]):
            metric_name = f"{metric_base}_{class_type}"
            
            print(f"Plotting {metric_name}")
            
            # Check if we have valid data
            valid_data_count = batch_df[metric_name].notna().sum()
            print(f"  Valid data points: {valid_data_count}/{len(batch_df)}")
            
            if valid_data_count == 0:
                print(f"  WARNING: No valid data for {metric_name}, skipping subplot")
                ax.text(0.5, 0.5, f"No data available\nfor {metric_name}", 
                       ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"Batch Size {batch_size}\n{class_type.capitalize()} Classes")
                continue
            
            print(f"  Range: [{batch_df[metric_name].min():.4f}, {batch_df[metric_name].max():.4f}]")
            
            # Create pivot table for heatmap
            pivot_data = batch_df.pivot_table(
                values=metric_name,
                index="eps",
                columns="lr",
                aggfunc="mean"
            )
            
            # Sort: epsilon descending, learning rate ascending
            pivot_data = pivot_data.sort_index(ascending=False)
            pivot_data = pivot_data.sort_index(axis=1, ascending=True)
            
            data = pivot_data.to_numpy(dtype=float, copy=True)
            nrows, ncols = data.shape
            eps_labels = [f"{e:.0e}" for e in pivot_data.index]
            lr_labels = [f"{lr:.0e}" for lr in pivot_data.columns]
            
            ax = axes[row_idx, col_idx]
            cmap = "YlOrRd_r" if "loss" in metric_base.lower() else "RdYlGn"
            
            im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=np.nanmin(data), vmax=np.nanmax(data))
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_name)
            
            # Annotate cells
            for i in range(nrows):
                for j in range(ncols):
                    val = data[i, j]
                    if np.isfinite(val):
                        norm_val = im.norm(val)
                        rgba = im.cmap(norm_val)
                        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        text_color = "white" if luminance < 0.5 else "black"
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                color=text_color, fontsize=8, weight="bold")
            
            # Title for each subplot
            if row_idx == 0:
                ax.set_title(f"Batch Size {batch_size}\nMajority Classes (10 common)", fontsize=10)
            else:
                ax.set_title(f"Minority Classes (10,230 rare)", fontsize=10)
            
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Epsilon")
            ax.set_xticks(range(ncols))
            ax.set_yticks(range(nrows))
            ax.set_xticklabels(lr_labels, rotation=45, ha="right")
            ax.set_yticklabels(eps_labels)
    
    plt.suptitle(
        f"AdaptiveSign {variant_name}: {metric_base} at Epoch {epoch}\nMajority vs Minority Classes",
        fontsize=14,
        y=0.995
    )
    plt.tight_layout()
    
    # Save figure
    save_dir = config.get_plots_directory() / "AdaptiveSign_ImbalancedMNIST_CNN" / "majority_minority"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{metric_base}_epoch{epoch}_{momentum_variant}_majority_minority.png"
    filepath = save_dir / filename
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {filepath}")
    
    plt.close(fig)


def plot_all_majority_minority(experiments, epoch=None):
    """Plot majority/minority comparisons for all key metrics."""
    metrics = [
        "va_Accuracy",
        "tr_Accuracy",
        "va_CrossEntropyLoss",
        "tr_CrossEntropyLoss",
    ]
    
    for momentum_variant in ["NM", "M"]:
        for metric in metrics:
            print(f"\n{'='*80}")
            print(f"Plotting {metric} ({momentum_variant})...")
            print(f"{'='*80}")
            try:
                plot_majority_minority_heatmap(
                    experiments,
                    metric_base=metric,
                    epoch=epoch,
                    momentum_variant=momentum_variant
                )
            except Exception as e:
                print(f"Error plotting {metric}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AdaptiveSign majority/minority stats")
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
        help="Specific metric to plot (e.g., va_Accuracy, tr_CrossEntropyLoss)"
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
        momentum_variants = ["M", "NM"] if args.momentum == "both" else [args.momentum]
        for variant in momentum_variants:
            plot_majority_minority_heatmap(
                exps_to_plot,
                metric_base=args.metric,
                epoch=args.epoch,
                momentum_variant=variant
            )
    else:
        # Plot all metrics
        plot_all_majority_minority(exps_to_plot, epoch=args.epoch)
