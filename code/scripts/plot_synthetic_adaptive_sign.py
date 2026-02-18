"""
Plotting script for Synthetic AdaptiveSign experiments.

Creates two types of plots:
1. Overall loss comparison: Best learning rate per optimizer, all on one plot
2. Per-tier breakdown: For each optimizer's best run, plot all 7 tier losses
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import wandb

from optexp import config


def load_experiment_data(project, group_name):
    """Load all runs from a specific experiment group."""
    api = wandb.Api()
    entity = config.get_wandb_entity()
    
    runs = api.runs(
        f"{entity}/{project}",
        filters={"group": group_name, "state": "finished"}
    )
    
    print(f"Found {len(runs)} finished runs in group '{group_name}'")
    
    data = []
    for run in runs:
        # Extract optimizer info
        opt_config = run.config.get("exp_config", {}).get("optim", {})
        opt_name = opt_config.get("__class__", "Unknown")
        
        # Extract learning rate
        lr_config = opt_config.get("learning_rate", {})
        if isinstance(lr_config, dict):
            lr_exp = lr_config.get("exponent", 0)
            lr_base = lr_config.get("base", 10)
            if isinstance(lr_exp, dict):
                num = float(lr_exp.get("numerator", 0))
                denom = float(lr_exp.get("denominator", 1))
                lr = float(lr_base) ** (num / denom)
            elif isinstance(lr_exp, str) and '/' in lr_exp:
                parts = lr_exp.split('/')
                num = float(parts[0])
                denom = float(parts[1])
                lr = float(lr_base) ** (num / denom)
            else:
                lr = float(lr_base) ** float(lr_exp)
        else:
            lr = None
        
        # Extract epsilon (for AdaptiveSign)
        eps = opt_config.get("eps", None)
        
        # Extract momentum
        momentum = opt_config.get("momentum", 0)
        if momentum is None:
            momentum = opt_config.get("beta1", 0)
        
        # Extract batch size
        batch_size = run.config.get("exp_config", {}).get("problem", {}).get("dataset", {}).get("batch_size", None)
        
        # Get history
        history = run.history()
        
        data.append({
            "run_id": run.id,
            "run_name": run.name,
            "opt": opt_name,
            "lr": lr,
            "eps": eps,
            "momentum": momentum,
            "batch_size": batch_size,
            "history": history,
        })
    
    return data


def find_best_runs_per_optimizer(data, metric="tr_CrossEntropyLoss", epoch=-1):
    """
    For each optimizer type, find the run with best (lowest) metric value.
    
    Groups by: (optimizer_name, epsilon, momentum, batch_size)
    Returns: Dict mapping optimizer key to best run data
    """
    best_runs = {}
    
    # Group runs by optimizer configuration
    for run_data in data:
        opt = run_data["opt"]
        eps = run_data["eps"]
        momentum = run_data["momentum"]
        batch_size = run_data["batch_size"]
        
        # Create key for this optimizer variant
        if opt == "AdaptiveSign":
            key = f"AdaptiveSign_eps{eps:.0e}_M{momentum:.1f}_bs{batch_size}"
        else:
            key = f"{opt}_M{momentum:.1f}_bs{batch_size}"
        
        # Get final metric value
        history = run_data["history"]
        if metric not in history.columns:
            continue
        
        metric_values = history[metric].dropna()
        if len(metric_values) == 0:
            continue
        
        final_value = metric_values.iloc[epoch]
        
        # Track best run for this optimizer variant
        if key not in best_runs or final_value < best_runs[key]["final_metric"]:
            best_runs[key] = {
                "run_data": run_data,
                "final_metric": final_value,
            }
    
    return best_runs


def plot_overall_loss_comparison(best_runs, metric="tr_CrossEntropyLoss", save_dir=None):
    """
    Plot overall loss for all best runs on one figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort by optimizer type for consistent colors
    sorted_keys = sorted(best_runs.keys())
    
    for key in sorted_keys:
        run_data = best_runs[key]["run_data"]
        history = run_data["history"]
        
        if metric not in history.columns:
            continue
        
        epochs = history["_step"].values
        values = history[metric].values
        
        # Create label
        opt = run_data["opt"]
        eps = run_data["eps"]
        lr = run_data["lr"]
        
        if opt == "AdaptiveSign":
            label = f"AdaptiveSign (ε={eps:.0e}, LR={lr:.0e})"
        else:
            label = f"{opt} (LR={lr:.0e})"
        
        ax.plot(epochs, values, label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Best Learning Rate per Optimizer: {metric}", fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"overall_comparison_{metric}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.close(fig)


def plot_tier_losses_per_optimizer(best_runs, tier_metric="tr_CrossEntropyLossByFrequencyTier", save_dir=None):
    """
    For each optimizer's best run, create a separate plot showing all 7 tier losses.
    """
    num_tiers = 7
    tier_labels = [
        "Tier 0 (64 samp/cls)",
        "Tier 1 (32 samp/cls)",
        "Tier 2 (16 samp/cls)",
        "Tier 3 (8 samp/cls)",
        "Tier 4 (4 samp/cls)",
        "Tier 5 (2 samp/cls)",
        "Tier 6 (1 samp/cls)",
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_tiers))
    
    for key, best_run_info in best_runs.items():
        run_data = best_run_info["run_data"]
        history = run_data["history"]
        
        # Check if tier metric exists
        if tier_metric not in history.columns:
            print(f"Warning: {tier_metric} not found for {key}")
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        epochs = history["_step"].values
        
        # Plot each tier
        for tier_idx in range(num_tiers):
            # Extract tier values from the list
            tier_values = []
            for val in history[tier_metric].values:
                if isinstance(val, (list, np.ndarray)) and len(val) > tier_idx:
                    tier_values.append(val[tier_idx])
                else:
                    tier_values.append(np.nan)
            
            tier_values = np.array(tier_values)
            
            ax.plot(epochs, tier_values, label=tier_labels[tier_idx], 
                   linewidth=2, alpha=0.8, color=colors[tier_idx])
        
        # Create title
        opt = run_data["opt"]
        eps = run_data["eps"]
        lr = run_data["lr"]
        batch_size = run_data["batch_size"]
        
        if opt == "AdaptiveSign":
            title = f"AdaptiveSign (ε={eps:.0e}, LR={lr:.0e}, BS={batch_size}): Loss per Tier"
        else:
            title = f"{opt} (LR={lr:.0e}, BS={batch_size}): Loss per Tier"
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(tier_metric, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize filename
            filename = key.replace(".", "p").replace(" ", "_")
            filepath = save_dir / f"tier_losses_{filename}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot synthetic AdaptiveSign experiment results")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="WandB project name (default: from config)"
    )
    parser.add_argument(
        "--group",
        type=str,
        default="AdaptiveSign_Synthetic_Linear",
        help="Experiment group name"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tr_CrossEntropyLoss",
        help="Metric for selecting best runs"
    )
    parser.add_argument(
        "--tier-metric",
        type=str,
        default="tr_CrossEntropyLossByFrequencyTier",
        help="Tier-specific metric to plot"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Filter by batch size (default: all batch sizes)"
    )
    
    args = parser.parse_args()
    
    project = args.project or config.get_wandb_project()
    
    print("="*70)
    print(f"Loading data from WandB")
    print("="*70)
    print(f"Project: {project}")
    print(f"Group: {args.group}")
    print()
    
    # Load data
    data = load_experiment_data(project, args.group)
    
    if len(data) == 0:
        print("ERROR: No data found")
        return
    
    # Filter by batch size if specified
    if args.batch_size is not None:
        data = [d for d in data if d["batch_size"] == args.batch_size]
        print(f"Filtered to {len(data)} runs with batch_size={args.batch_size}")
    
    # Find best runs per optimizer
    print(f"\nFinding best runs per optimizer (based on {args.metric})...")
    best_runs = find_best_runs_per_optimizer(data, metric=args.metric, epoch=-1)
    print(f"Found {len(best_runs)} optimizer variants")
    
    # Create save directory
    save_dir = Path(config.get_plots_directory()) / "AdaptiveSign_Synthetic_Linear"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot overall comparison
    print("\nCreating overall loss comparison plot...")
    plot_overall_loss_comparison(best_runs, metric=args.metric, save_dir=save_dir)
    
    # Plot tier losses per optimizer
    print("\nCreating per-optimizer tier loss plots...")
    plot_tier_losses_per_optimizer(best_runs, tier_metric=args.tier_metric, save_dir=save_dir)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
