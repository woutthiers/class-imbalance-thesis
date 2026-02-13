"""
Experiments for Adaptive Sign optimizer on Imbalanced MNIST dataset.

This script runs experiments comparing AdaptiveSign with different epsilon values
and learning rates on a realistic class-imbalanced dataset using a small CNN.

Dataset configuration:
- 10 common classes (original MNIST) with ~5000 samples each (50k total)
- ~10,240 rare classes (barcoded MNIST) with 5 samples each (~51k total)
- Total: ~10,250 classes, ~101k training samples

The experiment sweeps over:
- Learning rates (gamma): configurable range
- Epsilon values: [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
- Momentum variants: with and without momentum
"""
from optexp import Experiment, exp_runner_cli
from optexp.datasets.barcoded_mnist import (
    ImbalancedMNISTWithBarcodes,
    MNISTBarcodeOnly,
    MNISTAndBarcode,
)
from optexp.problems.imbalanced_classification import (
    FullBatchClassificationWithMajorityMinorityStats,
)
from optexp.models.cnn import SimpleMNISTCNN
from optexp.optimizers import (
    AdaptiveSign,
    AdaptiveSign_M,
    AdaptiveSign_NM,
    AdaptiveSignNormFirst_M,
    Adam_M,
    Adam_NM,
    SGD_M,
    SGD_NM,
)
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3, lr_grid

# ============================================================================
# CONFIGURATION - Adjust these parameters
# ============================================================================

# Batch sizes to test (will run experiments for each)
BATCH_SIZES = [64, 256, 1024]  # Three different batch sizes

# Learning rate grid - 4 learning rates
LR_START = -5    # 10^-5
LR_END = -2      # 10^-2
LR_DENSITY = 0   # 0=coarse gives 4 points: 10^-5, 10^-4, 10^-3, 10^-2

# Epsilon values to test - 4 epsilon values
EPSILON_VALUES = [1e-8, 1e-4, 1e-2, 1.0]

# Number of seeds (1 for quick exploration, 3 for robustness)
NUM_SEEDS = 1

# Training epochs
EPOCHS = 20

# Include baseline optimizers (Adam, SGD)
INCLUDE_BASELINES = False  # Disabled for quick focused comparison

# ============================================================================


def make_adaptive_sign_grid(epsilon_values, lr_start, lr_end, lr_density):
    """
    Create a grid of AdaptiveSign optimizers with different epsilon and learning rate values.
    
    Includes two momentum variants:
    - NM: No momentum (vanilla AdaptiveSign)
    - M: Standard momentum (normalize gradient first, then accumulate momentum)
    
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
        
        # With momentum (normalize then accumulate)
        for lr in learning_rates:
            optimizers.append(AdaptiveSign_M(lr, eps=eps))
    
    return optimizers


# Create AdaptiveSign optimizer grid
opts_adaptive_sign = make_adaptive_sign_grid(
    epsilon_values=EPSILON_VALUES,
    lr_start=LR_START,
    lr_end=LR_END,
    lr_density=LR_DENSITY,
)

# Create baseline optimizer grid (optional)
opts_baselines = []
if INCLUDE_BASELINES:
    # Adam baselines (typically good around 1e-3 to 1e-4)
    for lr in lr_grid(start=-4, end=-2, density=1):
        opts_baselines.append(Adam_NM(lr))
        opts_baselines.append(Adam_M(lr))
    
    # SGD baselines (typically good around 1e-1 to 1.0)
    for lr in lr_grid(start=-2, end=0, density=1):
        opts_baselines.append(SGD_NM(lr))
        opts_baselines.append(SGD_M(lr))

# Combine all optimizers
all_optimizers = opts_adaptive_sign + opts_baselines

# Select seeds
seeds = SEEDS_3[:NUM_SEEDS] if NUM_SEEDS <= 3 else SEEDS_3 + list(range(3, NUM_SEEDS))

group = "AdaptiveSign_ImbalancedMNIST_CNN"

# Generate experiments for each batch size
experiments = []
for batch_size in BATCH_SIZES:
    # Create dataset with specific batch size
    dataset = ImbalancedMNISTWithBarcodes(name="MNIST", batch_size=batch_size)
    model = SimpleMNISTCNN()
    # Use grouped metrics: tracks performance on 10 majority classes vs 10,240 minority classes
    problem = FullBatchClassificationWithMajorityMinorityStats(
        model, dataset, num_majority_classes=10
    )
    
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

# SLURM configuration for VSC
SLURM_CONFIG = slurm_config.VSC_CPU_4H

# For testing, use shorter time:
# SLURM_CONFIG = slurm_config.VSC_CPU_1H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
