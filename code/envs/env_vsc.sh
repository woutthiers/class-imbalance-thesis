#!/bin/bash
# VSC-specific environment variables
# Copy this file and customize with your settings
# Run: source code/envs/env_vsc.sh

# Workspace directory (use VSC_DATA for persistent storage)
export OPTEXP_WORKSPACE=$VSC_DATA/optexp_workspace

# Create workspace directory if it doesn't exist
mkdir -p $OPTEXP_WORKSPACE/{datasets,experiments,plots,tokenizers,wandb_cache}

# Wandb configuration
export OPTEXP_WANDB_ENABLED=true
export OPTEXP_WANDB_MODE=offline  # Set to "online" if you want to sync to wandb
export OPTEXP_WANDB_PROJECT=class-imbalance-experiments
export OPTEXP_WANDB_ENTITY=your_wandb_username  # CHANGE THIS to your wandb username
export WANDB_API_KEY=your_wandb_api_key_here    # CHANGE THIS to your actual API key

# VSC SLURM configuration
export OPTEXP_SLURM_NOTIFICATION_EMAIL=your.email@kuleuven.be  # CHANGE THIS
export OPTEXP_SLURM_ACCOUNT=lp_your_project                     # CHANGE THIS to your VSC project account

# Optional: Set cluster explicitly
export OPTEXP_SLURM_CLUSTER=wice
export OPTEXP_SLURM_PARTITION=gpu

echo "Environment variables set:"
echo "  OPTEXP_WORKSPACE: $OPTEXP_WORKSPACE"
echo "  OPTEXP_WANDB_PROJECT: $OPTEXP_WANDB_PROJECT"
echo "  OPTEXP_SLURM_ACCOUNT: $OPTEXP_SLURM_ACCOUNT"
echo "  OPTEXP_SLURM_CLUSTER: $OPTEXP_SLURM_CLUSTER"
