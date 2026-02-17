"""
Download data from crashed WandB runs that still have logged epochs.

This script downloads data from runs marked as "crashed" in WandB but that
still have partial data logged (e.g., 4 epochs before crashing).

Usage:
    python scripts/download_crashed_runs.py
"""
import wandb
import pandas as pd
from pathlib import Path
from optexp import config
from optexp.experiments.vision.barcoded_mnist_adaptive_sign import experiments

def download_crashed_run_data(run, save_dir):
    """Download history from a crashed run."""
    print(f"  Downloading {run.name} ({run.state}, {len(run.history())} steps)...")
    
    # Get run history
    history = run.history()
    
    if len(history) == 0:
        print(f"    WARNING: No data in history")
        return None
    
    # Save to CSV
    run_id = run.id
    filepath = save_dir / f"{run_id}.csv"
    history.to_csv(filepath, index=False)
    print(f"    Saved {len(history)} rows to {filepath}")
    
    return history

def main():
    print("="*70)
    print("Downloading Crashed Run Data from WandB")
    print("="*70)
    
    # Get WandB project info from first experiment
    if not experiments:
        print("ERROR: No experiments found")
        return
    
    first_exp = experiments[0]
    group = first_exp.group
    
    print(f"\nGroup: {group}")
    print(f"Looking for crashed runs with batch_size=8...")
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Get entity and project from config
    entity = config.get_wandb_entity()
    project = config.get_wandb_project()
    
    print(f"Entity: {entity}")
    print(f"Project: {project}")
    
    # Fetch all runs from this group
    runs = api.runs(
        f"{entity}/{project}",
        filters={"group": group}
    )
    
    print(f"\nTotal runs in group: {len(runs)}")
    
    # Filter for crashed runs with batch_size=8
    crashed_bs8_runs = []
    for run in runs:
        if run.state == "crashed" and run.config.get("batch_size") == 8:
            history_len = len(run.history())
            if history_len > 0:
                crashed_bs8_runs.append(run)
                print(f"  Found: {run.name} - {history_len} epochs logged")
    
    print(f"\nFound {len(crashed_bs8_runs)} crashed batch_size=8 runs with data")
    
    if len(crashed_bs8_runs) == 0:
        print("No crashed runs with data found. Nothing to download.")
        return
    
    # Create save directory
    save_dir = config.get_data_directory() / "crashed_runs"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {save_dir}")
    
    # Download each crashed run
    print("\nDownloading crashed runs:")
    for run in crashed_bs8_runs:
        download_crashed_run_data(run, save_dir)
    
    print("\n" + "="*70)
    print("Download complete!")
    print("="*70)
    print(f"\nData saved to: {save_dir}")
    print("\nNote: This data is separate from the normal experiment data.")
    print("You may need to manually merge it or create custom plotting scripts.")

if __name__ == "__main__":
    main()
