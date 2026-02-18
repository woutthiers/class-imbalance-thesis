"""
Plot heatmaps for crashed batch_size=8 runs using downloaded CSV data.

This script creates heatmaps from the crashed run data that was downloaded
separately using download_crashed_runs.py.

Usage:
    python scripts/plot_crashed_runs_heatmap.py [--epoch EPOCH] [--metric METRIC]
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from optexp import config


def load_crashed_run_metadata(run_id, api):
    """Load run metadata from WandB to get experiment config."""
    entity = config.get_wandb_entity()
    project = config.get_wandb_project()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        exp_config = run.config.get("exp_config", {})
        
        # Extract optimizer info
        optim_config = exp_config.get("optim", {})
        opt_name = optim_config.get("__class__", "Unknown")
        
        # Extract learning rate
        lr_config = optim_config.get("learning_rate", {})
        if isinstance(lr_config, dict):
            lr_exp = lr_config.get("exponent", 0)
            lr_base = lr_config.get("base", 10)
            # Handle fraction representation (dict, string, or number)
            if isinstance(lr_exp, dict):
                num = float(lr_exp.get("numerator", 0))
                denom = float(lr_exp.get("denominator", 1))
                lr = float(lr_base) ** (num / denom)
            elif isinstance(lr_exp, str) and '/' in lr_exp:
                # Handle string fractions like '-7/2'
                parts = lr_exp.split('/')
                num = float(parts[0])
                denom = float(parts[1])
                lr = float(lr_base) ** (num / denom)
            else:
                lr = float(lr_base) ** float(lr_exp)
        else:
            lr = None
        
        # Extract epsilon (for AdaptiveSign)
        eps = optim_config.get("eps", None)
        
        # Extract momentum
        momentum = optim_config.get("momentum", 0)
        if momentum is None:
            momentum = optim_config.get("beta1", 0)
        
        # Extract batch size
        batch_size = exp_config.get("problem", {}).get("dataset", {}).get("batch_size", None)
        
        return {
            "opt": opt_name,
            "lr": lr,
            "eps": eps,
            "momentum": momentum,
            "batch_size": batch_size,
            "run_id": run_id,
            "run_name": run.name,
        }
    except Exception as e:
        print(f"Warning: Could not load metadata for run {run_id}: {e}")
        return None


def load_all_crashed_runs(crashed_dir, api):
    """Load all crashed run CSVs and their metadata."""
    csv_files = list(crashed_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    
    for csv_file in csv_files:
        run_id = csv_file.stem
        
        # Load CSV
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            continue
        
        # Load metadata
        metadata = load_crashed_run_metadata(run_id, api)
        
        if metadata is None:
            continue
        
        # Add metadata to each row
        for col, val in metadata.items():
            df[col] = val
        
        all_data.append(df)
        
        if len(all_data) % 10 == 0:
            print(f"  Loaded {len(all_data)} runs...")
    
    if not all_data:
        print("ERROR: No valid data loaded")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Unique runs: {combined_df['run_id'].nunique()}")
    
    return combined_df


def plot_crashed_runs_heatmap(
    crashed_dir,
    metric_name="va_Accuracy",
    epoch=None,
    momentum_variant="M",
):
    """
    Create heatmap for crashed batch_size=8 runs.
    
    Args:
        crashed_dir: Directory with crashed run CSVs
        metric_name: Metric to visualize
        epoch: Which epoch to plot (default: max available per run)
        momentum_variant: "M" for with momentum, "NM" for without
    """
    print("="*70)
    print(f"Plotting Crashed Runs: {metric_name}")
    print("="*70)
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Load all data
    print("\nLoading crashed run data...")
    df = load_all_crashed_runs(crashed_dir, api)
    
    if df is None or len(df) == 0:
        print("ERROR: No data to plot")
        return
    
    print(f"\n=== Data Summary ===")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique optimizers: {df['opt'].unique()}")
    print(f"Unique batch_sizes: {df['batch_size'].unique()}")
    print(f"Epoch range: {df['epoch'].min():.0f} - {df['epoch'].max():.0f}")
    
    # Filter for AdaptiveSign only
    df = df[df["opt"] == "AdaptiveSign"]
    print(f"\nAfter filtering for AdaptiveSign: {len(df)} rows, {df['run_id'].nunique()} runs")
    
    # Filter by momentum
    if momentum_variant == "M":
        df = df[df["momentum"] > 0]
        variant_name = "with Momentum"
    else:
        df = df[df["momentum"] == 0]
        variant_name = "without Momentum"
    
    print(f"After filtering for {variant_name}: {len(df)} rows, {df['run_id'].nunique()} runs")
    
    # Handle epoch selection
    if epoch is None:
        # Use last available epoch for each run
        print("\nUsing last available epoch for each run")
        df = df.loc[df.groupby('run_id')['epoch'].idxmax()]
    else:
        # Use specific epoch, or last available if not reached
        print(f"\nUsing epoch {epoch} (or last available)")
        epoch_df = []
        for run_id in df['run_id'].unique():
            run_df = df[df['run_id'] == run_id]
            if epoch in run_df['epoch'].values:
                epoch_df.append(run_df[run_df['epoch'] == epoch])
            else:
                # Use last available
                last_epoch = run_df['epoch'].max()
                epoch_df.append(run_df[run_df['epoch'] == last_epoch])
        df = pd.concat(epoch_df, ignore_index=True)
    
    print(f"\nFinal data for plotting: {len(df)} rows")
    print(f"LR range: {df['lr'].min():.2e} - {df['lr'].max():.2e}")
    print(f"Epsilon range: {df['eps'].min():.2e} - {df['eps'].max():.2e}")
    
    # Check if metric exists
    if metric_name not in df.columns:
        print(f"\nERROR: Metric '{metric_name}' not found in data")
        print(f"Available metrics: {[c for c in df.columns if 'va_' in c or 'tr_' in c]}")
        return
    
    # Create pivot table
    pivot_data = df.pivot_table(
        values=metric_name,
        index="eps",
        columns="lr",
        aggfunc="mean"
    )
    
    # Sort
    pivot_data = pivot_data.sort_index(ascending=False)
    pivot_data = pivot_data.sort_index(axis=1, ascending=True)
    
    print(f"\nPivot table shape: {pivot_data.shape}")
    print(f"Epsilon values: {list(pivot_data.index)}")
    print(f"Learning rates: {list(pivot_data.columns)}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = pivot_data.to_numpy(dtype=float, copy=True)
    nrows, ncols = data.shape
    
    eps_labels = [f"{e:.0e}" for e in pivot_data.index]
    lr_labels = [f"{lr:.0e}" for lr in pivot_data.columns]
    
    cmap = "YlOrRd_r" if "loss" in metric_name.lower() else "RdYlGn"
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    
    # Colorbar
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
                        color=text_color, fontsize=10, weight="bold")
    
    ax.set_title(f"Batch Size 8 (Crashed Runs)\nAdaptiveSign {variant_name}: {metric_name}", fontsize=14)
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Epsilon", fontsize=12)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels(lr_labels, rotation=45, ha="right")
    ax.set_yticklabels(eps_labels)
    
    plt.tight_layout()
    
    # Save
    save_dir = Path(config.get_plots_directory()) / "AdaptiveSign_ImbalancedMNIST_CNN" / "crashed_runs"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_str = f"epoch{epoch}" if epoch else "last_epoch"
    filename = f"{metric_name}_{epoch_str}_{momentum_variant}_batch8.png"
    filepath = save_dir / filename
    
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot crashed batch_size=8 heatmaps")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to plot (default: last available per run)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="va_Accuracy",
        help="Metric to plot (default: va_Accuracy)"
    )
    parser.add_argument(
        "--momentum",
        type=str,
        choices=["M", "NM", "both"],
        default="both",
        help="Momentum variant (default: both)"
    )
    
    args = parser.parse_args()
    
    # Get crashed runs directory
    crashed_dir = Path(config.get_dataset_directory()) / "data" / "crashed_runs"
    
    if not crashed_dir.exists():
        print(f"ERROR: Crashed runs directory not found: {crashed_dir}")
        print("Run download_crashed_runs.py first")
        exit(1)
    
    # Plot
    momentum_variants = ["M", "NM"] if args.momentum == "both" else [args.momentum]
    
    for variant in momentum_variants:
        plot_crashed_runs_heatmap(
            crashed_dir,
            metric_name=args.metric,
            epoch=args.epoch,
            momentum_variant=variant
        )
