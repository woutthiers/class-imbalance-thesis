"""
Custom plotting script for AdaptiveSign experiments with batch size comparison.

Creates grid plots with separate colors for each batch size, comparing:
- Different epsilon values (subplots)
- Different learning rates (x-axis)
- Different batch sizes (different colored lines)
- Different momentum variants (separate plots or lines)

Usage:
    python plot_adaptive_sign_results.py --epoch 20
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from optexp import config
from optexp.plotter.data_utils import get_exps_data_epoch, load_data_for_exps
from optexp.plotter.names_and_consts import displayname

# Import the experiments
import sys
sys.path.insert(0, str(Path(__file__).parent / "code" / "src"))
from optexp.experiments.vision.barcoded_mnist_adaptive_sign import experiments


def plot_adaptive_sign_with_batch_sizes(experiments_list, plotting_epoch=20):
    """
    Create grid plots showing learning rate performance across batch sizes.
    
    Creates a grid where:
    - Each row = different epsilon value
    - X-axis = learning rate
    - Y-axis = performance metric (accuracy or loss)
    - Different colors = different batch sizes
    - Different line styles = different momentum variants
    """
    
    # Load experiment data
    print(f"Loading data from {len(experiments_list)} experiments...")
    exps_w_data = load_data_for_exps(experiments_list)
    
    # Metrics to plot
    metric_names = ['tr_loss', 'tr_acc', 'va_loss', 'va_acc']
    
    # Get data at specific epoch
    exps_df = get_exps_data_epoch(exps_w_data, metric_names, plotting_epoch, using_step=False)
    
    if exps_df.empty:
        print("No data found! Make sure experiments have completed and data is downloaded.")
        return
    
    print(f"Loaded {len(exps_df)} experiment results")
    print(f"Columns: {exps_df.columns.tolist()}")
    
    # Extract batch size from experiment problem
    def get_batch_size(exp):
        return exp.problem.dataset.batch_size
    
    # Add batch size column
    batch_sizes = []
    for exp in exps_w_data:
        batch_sizes.append(get_batch_size(exp['experiment']))
    exps_df['batch_size'] = batch_sizes
    
    # Extract epsilon from optimizer
    def get_epsilon(exp):
        opt = exp.optimizer
        if hasattr(opt, 'eps'):
            return opt.eps
        return None
    
    epsilons = []
    for exp in exps_w_data:
        epsilons.append(get_epsilon(exp['experiment']))
    exps_df['epsilon'] = epsilons
    
    # Get unique values
    unique_batch_sizes = sorted(exps_df['batch_size'].unique())
    unique_epsilons = sorted(exps_df['epsilon'].unique())
    unique_momentums = exps_df['momentum'].unique()
    
    print(f"\nBatch sizes: {unique_batch_sizes}")
    print(f"Epsilons: {unique_epsilons}")
    print(f"Momentum types: {unique_momentums}")
    
    # Color scheme for batch sizes
    batch_colors = {
        64: '#1f77b4',    # blue
        256: '#ff7f0e',   # orange
        1024: '#2ca02c',  # green
    }
    
    # Line styles for momentum variants
    momentum_styles = {
        False: '-',       # no momentum: solid
        True: '--',       # momentum: dashed
        'NormFirst': ':',  # normalize-first: dotted
    }
    
    # Determine momentum type from optimizer name
    def get_momentum_type(opt_name):
        if 'NormFirst' in opt_name:
            return 'NormFirst'
        elif opt_name.endswith('_M'):
            return True
        elif opt_name.endswith('_NM'):
            return False
        return None
    
    momentum_types = []
    for exp in exps_w_data:
        opt_name = exp['experiment'].optimizer.__class__.__name__
        momentum_types.append(get_momentum_type(opt_name))
    exps_df['momentum_type'] = momentum_types
    
    unique_momentum_types = [m for m in exps_df['momentum_type'].unique() if m is not None]
    print(f"Momentum types found: {unique_momentum_types}")
    
    # Create save directory
    group = experiments_list[0].group
    save_path = config.get_plots_directory() / Path(group) / Path("batch_comparison")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot for each metric
    for metric in metric_names:
        if metric not in exps_df.columns:
            continue
        
        print(f"\nPlotting {metric}...")
        
        # Create subplot grid: one row per epsilon value
        n_eps = len(unique_epsilons)
        fig, axes = plt.subplots(n_eps, 1, figsize=(10, 3*n_eps), squeeze=False)
        
        for i, epsilon in enumerate(unique_epsilons):
            ax = axes[i, 0]
            
            # Plot each combination of batch size and momentum type
            for batch_size in unique_batch_sizes:
                for momentum_type in unique_momentum_types:
                    # Filter data
                    mask = (
                        (exps_df['epsilon'] == epsilon) &
                        (exps_df['batch_size'] == batch_size) &
                        (exps_df['momentum_type'] == momentum_type)
                    )
                    subset = exps_df[mask]
                    
                    if subset.empty:
                        continue
                    
                    # Group by learning rate and aggregate across seeds
                    grouped = subset.groupby('lr')[metric].agg(['mean', 'min', 'max'])
                    
                    if grouped.empty:
                        continue
                    
                    lrs = grouped.index.values
                    
                    # Determine label and style
                    mom_label = {
                        False: 'No Mom',
                        True: 'Mom',
                        'NormFirst': 'NormFirst'
                    }.get(momentum_type, str(momentum_type))
                    
                    label = f"BS={batch_size}, {mom_label}"
                    color = batch_colors.get(batch_size, 'gray')
                    linestyle = momentum_styles.get(momentum_type, '-')
                    
                    # Plot mean with min/max shading
                    ax.plot(lrs, grouped['mean'], label=label, color=color, 
                           linestyle=linestyle, marker='o', markersize=4)
                    ax.fill_between(lrs, grouped['min'], grouped['max'], 
                                   color=color, alpha=0.2)
            
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel(displayname(metric))
            ax.set_title(f'ε = {epsilon}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            
            if 'loss' in metric.lower():
                ax.set_yscale('log')
        
        fig.suptitle(f'{displayname(metric)} at Epoch {plotting_epoch}\n'
                    f'Grid: LR vs Epsilon, Colors = Batch Size, Lines = Momentum',
                    fontsize=12)
        fig.tight_layout()
        
        # Save
        filepath = save_path / f"{metric}_epoch{plotting_epoch}_by_batch.pdf"
        plt.savefig(filepath)
        print(f"Saved: {filepath}")
        
        filepath_png = save_path / f"{metric}_epoch{plotting_epoch}_by_batch.png"
        plt.savefig(filepath_png, dpi=150)
        print(f"Saved: {filepath_png}")
        
        plt.close(fig)
    
    print(f"\n✓ All plots saved to: {save_path}")


def plot_heatmap_comparison(experiments_list, plotting_epoch=20):
    """
    Create heatmap plots showing epsilon vs learning rate for each batch size.
    """
    print("\nCreating heatmap visualizations...")
    
    # Load data
    exps_w_data = load_data_for_exps(experiments_list)
    metric_names = ['tr_acc', 'va_acc']
    exps_df = get_exps_data_epoch(exps_w_data, metric_names, plotting_epoch, using_step=False)
    
    if exps_df.empty:
        return
    
    # Add batch size and epsilon
    batch_sizes = [exp['experiment'].problem.dataset.batch_size for exp in exps_w_data]
    exps_df['batch_size'] = batch_sizes
    
    epsilons = []
    for exp in exps_w_data:
        opt = exp['experiment'].optimizer
        epsilons.append(opt.eps if hasattr(opt, 'eps') else None)
    exps_df['epsilon'] = epsilons
    
    # Filter to just no-momentum for simplicity
    exps_df = exps_df[exps_df['momentum'] == False]
    
    unique_batch_sizes = sorted(exps_df['batch_size'].unique())
    
    group = experiments_list[0].group
    save_path = config.get_plots_directory() / Path(group) / Path("batch_comparison")
    
    for metric in metric_names:
        if metric not in exps_df.columns:
            continue
        
        # Create subplot for each batch size
        fig, axes = plt.subplots(1, len(unique_batch_sizes), 
                                figsize=(5*len(unique_batch_sizes), 4), squeeze=False)
        
        for i, batch_size in enumerate(unique_batch_sizes):
            ax = axes[0, i]
            
            # Filter to this batch size
            subset = exps_df[exps_df['batch_size'] == batch_size]
            
            # Pivot to create heatmap
            pivot = subset.pivot_table(values=metric, index='epsilon', 
                                      columns='lr', aggfunc='mean')
            
            if not pivot.empty:
                im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([f'{lr:.0e}' for lr in pivot.columns], rotation=45)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([f'{eps:.0e}' for eps in pivot.index])
                ax.set_xlabel('Learning Rate')
                ax.set_ylabel('Epsilon')
                ax.set_title(f'Batch Size = {batch_size}')
                plt.colorbar(im, ax=ax, label=displayname(metric))
        
        fig.suptitle(f'{displayname(metric)} Heatmap at Epoch {plotting_epoch}')
        fig.tight_layout()
        
        filepath = save_path / f"{metric}_epoch{plotting_epoch}_heatmap.pdf"
        plt.savefig(filepath)
        print(f"Saved heatmap: {filepath}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot AdaptiveSign experiment results')
    parser.add_argument('--epoch', type=int, default=20, 
                       help='Which epoch to plot (default: 20)')
    parser.add_argument('--heatmap', action='store_true',
                       help='Also create heatmap visualizations')
    
    args = parser.parse_args()
    
    print(f"Plotting results for {len(experiments)} experiments at epoch {args.epoch}")
    
    # Create standard grid plots with batch size colors
    plot_adaptive_sign_with_batch_sizes(experiments, plotting_epoch=args.epoch)
    
    # Optionally create heatmaps
    if args.heatmap:
        plot_heatmap_comparison(experiments, plotting_epoch=args.epoch)
    
    print("\n✓ Done!")
