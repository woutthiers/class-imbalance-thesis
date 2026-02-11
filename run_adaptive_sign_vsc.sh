#!/bin/bash
#SBATCH --job-name=adaptive_sign_mnist
#SBATCH --output=logs/adaptive_sign_%A_%a.out
#SBATCH --error=logs/adaptive_sign_%A_%a.err
#SBATCH --array=0-99%20
#SBATCH --time=0-04:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --account=lp_edu_mlopt_2024

# This is a SLURM job submission script for VSC wice cluster
# Adjust the --array parameter based on the number of experiments
# Format: --array=0-N%M where N is (number of experiments - 1) and M is max parallel jobs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules and activate environment
source vsc_setup.sh
cd code
source envs/env_vsc.sh

# Run the experiment
# The experiment runner will use SLURM_ARRAY_TASK_ID to determine which experiment to run
python -m optexp.experiments.vision.barcoded_mnist_adaptive_sign \
    --run-id $SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
