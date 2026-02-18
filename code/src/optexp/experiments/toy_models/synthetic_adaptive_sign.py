"""
Experiments for Adaptive Sign optimizer on Synthetic Imbalanced Dataset.

This script runs experiments comparing AdaptiveSign with different epsilon values
and learning rates on a synthetic class-imbalanced dataset with a linear classifier.

Dataset configuration:
- Synthetic data with step-function class imbalance (size=7):
  - 128 classes total
  - 1 class with 64 samples (most frequent)
  - 2 classes with 32 samples
  - 4 classes with 16 samples
  - 8 classes with 8 samples
  - 16 classes with 4 samples
  - 32 classes with 2 samples
  - 64 classes with 1 sample (least frequent)
  - Total: 896 samples
  - Imbalance ratio: 64:1

The experiment sweeps over:
- Learning rates (gamma): 7 learning rates from 10^-7 to 10^-1
- Epsilon values: [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
- Batch sizes: [32, 128, 896] (including full batch)
- Momentum variants: with and without momentum
"""
from optexp import MLP, Experiment, exp_runner_cli
from optexp.datasets.synthetic_dataset import GaussianImbalancedY
from optexp.optimizers import (
    AdaptiveSign_M,
    AdaptiveSign_NM,
    Adam_M,
    Adam_NM,
    SGD_M,
    SGD_NM,
)
from optexp.problems.frequency_tier_classification import ClassificationWithFrequencyTierStats
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3, lr_grid
from optexp.optimizers.learning_rate import LearningRate

# ============================================================================
# CONFIGURATION - Adjust these parameters
# ============================================================================

# Toggle between full grid search and focused comparison
USE_FOCUSED_COMPARISON = False  # Set to True for quick focused test

# === FULL GRID SEARCH CONFIGURATION ===
# Batch sizes to test
BATCH_SIZES = [128]  # Small, medium, full batch

# Learning rate grid - 7 learning rates
LR_START = -7   # 10^-7
LR_END = -1     # 10^-1
LR_DENSITY = 0   # 0=coarse gives 7 points

# Epsilon values to test - 5 epsilon values
EPSILON_VALUES = [1e-8]

# Number of seeds
NUM_SEEDS = 1

# Training epochs
EPOCHS = 50

# Include baseline optimizers (Adam, SGD)
INCLUDE_BASELINES = False

# === FOCUSED COMPARISON CONFIGURATION ===
# SGD (LR=3e-4) vs AdaptiveSign (eps=1e-8, LR=1e-4)
# BATCH_SIZES = [896]  # Only full batch
# NUM_SEEDS = 1
# EPOCHS = 2000
# INCLUDE_BASELINES = False

# ============================================================================

# Dataset: Synthetic Gaussian data with imbalanced classes (size=7)
# - 896 samples, 128 classes, 1024 dimensions
# - Imbalance ratio: 64:1 (most frequent: 64 samples, least: 1 sample)
# - 7 frequency tiers with class boundaries: [1, 3, 7, 15, 31, 63, 127]
DATASET_SIZE = 7

# Tier boundaries for size=7 synthetic dataset
# Tier 0: classes 0-0 (1 class × 64 samples)
# Tier 1: classes 1-2 (2 classes × 32 samples)
# Tier 2: classes 3-6 (4 classes × 16 samples)
# Tier 3: classes 7-14 (8 classes × 8 samples)
# Tier 4: classes 15-30 (16 classes × 4 samples)
# Tier 5: classes 31-62 (32 classes × 2 samples)
# Tier 6: classes 63-126 (64 classes × 1 sample)
TIER_BOUNDARIES = [1, 3, 7, 15, 31, 63, 127]


def make_adaptive_sign_grid(epsilon_values, lr_start, lr_end, lr_density):
    """
    Create a grid of AdaptiveSign optimizers with different epsilon and learning rate values.
    
    Includes two momentum variants:
    - NM: No momentum (vanilla AdaptiveSign)
    - M: Standard momentum (first calculate momentum, then normalize)
    
    Args:
        epsilon_values: List of epsilon values to test
        lr_start: Starting exponent for learning rate (10^lr_start)
        lr_end: Ending exponent for learning rate (10^lr_end)
        lr_density: Density of learning rate grid (0=coarse, 1=medium, 2=fine)
    
    Returns:
        List of optimizer configurations
    """
    optimizers = []
    learning_rates = lr_grid(start=lr_start, end=lr_end, density=lr_density)
    
    for eps in epsilon_values:
        # Without momentum
        for lr in learning_rates:
            optimizers.append(AdaptiveSign_NM(lr, eps=eps))
        
        # With momentum (first calculate momentum, then normalize)
        for lr in learning_rates:
            optimizers.append(AdaptiveSign_M(lr, eps=eps))
    
    return optimizers


# Create optimizer configurations based on mode
if USE_FOCUSED_COMPARISON:
    # Focused comparison: SGD vs AdaptiveSign with heavy normalization
    opts_adaptive_sign = [
        AdaptiveSign_NM(LearningRate(base=10, exponent=-4), eps=1e-8),
    ]
    opts_baselines = [
        SGD_NM(LearningRate(base=10, exponent=-3.5)),  # 3e-4
    ]
else:
    # Full grid search
    opts_adaptive_sign = make_adaptive_sign_grid(
        epsilon_values=EPSILON_VALUES,
        lr_start=LR_START,
        lr_end=LR_END,
        lr_density=LR_DENSITY,
    )
    
    opts_baselines = []
    if INCLUDE_BASELINES:
        # Adam baselines (minimal grid: 3 learning rates)
        for lr in lr_grid(start=-4, end=-2, density=0):
            opts_baselines.append(Adam_NM(lr))
            opts_baselines.append(Adam_M(lr))
        
        # SGD baselines
        for lr in lr_grid(start=-4, end=-1, density=0):
            opts_baselines.append(SGD_NM(lr))
            opts_baselines.append(SGD_M(lr))

# Combine all optimizers
all_optimizers = opts_adaptive_sign + opts_baselines

# Select seeds
seeds = SEEDS_3[:NUM_SEEDS] if NUM_SEEDS <= 3 else SEEDS_3 + list(range(3, NUM_SEEDS))

group = "AdaptiveSign_Synthetic_Linear"

# Model: Linear classifier (logistic regression)
model = MLP(hidden_layers=None, activation=None)

# Generate experiments for each batch size
experiments = []
for batch_size in BATCH_SIZES:
    # Create dataset with specific batch size (mean=1 to avoid orthogonality)
    dataset = GaussianImbalancedY(batch_size=batch_size, size=DATASET_SIZE, x_mean=1.0)
    
    # Use frequency tier tracking (7 tiers for size=7)
    problem = ClassificationWithFrequencyTierStats(model, dataset, tier_boundaries=TIER_BOUNDARIES)
    
    # Generate experiments for this batch size
    batch_experiments = Experiment.generate_experiments_from_opts_and_seeds(
        opts_and_seeds=[(all_optimizers, seeds)],
        problem=problem,
        epochs=EPOCHS,
        group=group,
    )
    
    experiments.extend(batch_experiments)
    print(f"Generated {len(batch_experiments)} experiments for batch_size={batch_size}")

print(f"\nTotal experiments: {len(experiments)}")
print(f"  - Batch sizes: {BATCH_SIZES}")
print(f"  - Optimizers per batch size: {len(all_optimizers)}")
print(f"  - Seeds: {NUM_SEEDS}")
print(f"  - AdaptiveSign configs: {len(opts_adaptive_sign)}")
print(f"  - Baseline configs: {len(opts_baselines)}")
print(f"\nDataset info:")
print(f"  - Size parameter: {DATASET_SIZE}")
print(f"  - Total samples: {DATASET_SIZE * 2**DATASET_SIZE}")
print(f"  - Total classes: {2**DATASET_SIZE - 1}")
print(f"  - Dimensionality: {(DATASET_SIZE + 1) * 2**DATASET_SIZE}")
print(f"  - Imbalance ratio: {2**(DATASET_SIZE-1)}:1")

# SLURM configuration for VSC CPU
SLURM_CONFIG = slurm_config.VSC_CPU_4H

# For faster testing:
# SLURM_CONFIG = slurm_config.VSC_CPU_1H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
