# VSC Configuration Files

This directory contains the necessary configuration files for running the class-imbalance experiments on the VSC wice cluster.

## Files Created

1. **`vsc_setup.sh`** (in root directory)
   - Loads required modules (Python, CUDA, PyTorch)
   - Creates and activates virtual environment
   - Run this first with: `source vsc_setup.sh`

2. **`code/envs/env_vsc.sh`**
   - Sets environment variables for VSC
   - Configure your W&B credentials and VSC account here
   - Run after setup with: `source code/envs/env_vsc.sh`

3. **`test_vsc_setup.sh`** (in root directory)
   - SLURM job script to test your setup
   - Edit the account and email, then run: `sbatch test_vsc_setup.sh`

4. **`VSC_ONDEMAND_GUIDE.md`** (in root directory)
   - Complete guide for using VSC OnDemand with VS Code
   - Step-by-step instructions from cloning to running experiments

## Quick Start on VSC OnDemand

### 1. Open VS Code on OnDemand
- Go to https://login.hpc.kuleuven.be/
- Interactive Apps → VS Code
- Select wice cluster, 4 cores, 8GB RAM, 2 hours

### 2. Clone and Setup
```bash
cd $VSC_DATA
git clone https://github.com/fKunstner/class-imbalance-sgd-adam.git
cd class-imbalance-sgd-adam
source vsc_setup.sh
```

### 3. Install (First Time Only)
```bash
cd code
pip install -r requirements/main.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torchtext==0.14.1 portalocker==2.7.0 lightning==2.0.9 torchdata==0.5.1
pip install -e .
```

### 4. Configure Environment
Edit `code/envs/env_vsc.sh` with your details:
- Your W&B username and API key
- Your VSC email and project account (e.g., `lp_my_project`)

Then load:
```bash
source code/envs/env_vsc.sh
```

### 5. Run Experiments
```bash
cd src/optexp/experiments/toy_models

# Test locally (quick)
python balanced_x_perclass.py --single 0

# Submit to cluster (for longer runs)
python balanced_x_perclass.py --slurm
```

## VSC-Specific SLURM Configurations

The following SLURM configs have been added for VSC wice (A100 GPUs):

```python
from optexp.runner.slurm import slurm_config

# Short experiments (testing)
slurm_config.VSC_A100_15MIN  # 15 min, 8GB, 2 CPUs, 1 A100
slurm_config.VSC_A100_30MIN  # 30 min, 8GB, 2 CPUs, 1 A100

# Medium experiments
slurm_config.VSC_A100_1H     # 1 hour, 16GB, 4 CPUs, 1 A100
slurm_config.VSC_A100_2H     # 2 hours, 16GB, 4 CPUs, 1 A100
slurm_config.VSC_A100_4H     # 4 hours, 32GB, 8 CPUs, 1 A100

# Long experiments
slurm_config.VSC_A100_8H     # 8 hours, 32GB, 8 CPUs, 1 A100
slurm_config.VSC_A100_12H    # 12 hours, 32GB, 8 CPUs, 1 A100
slurm_config.VSC_A100_24H    # 24 hours, 64GB, 16 CPUs, 1 A100

# Multi-GPU experiments
slurm_config.VSC_2_A100_12H  # 12 hours, 64GB, 16 CPUs, 2 A100s
slurm_config.VSC_2_A100_24H  # 24 hours, 128GB, 32 CPUs, 2 A100s
slurm_config.VSC_4_A100_24H  # 24 hours, 256GB, 64 CPUs, 4 A100s
```

Use them in your experiment files:
```python
SLURM_CONFIG = slurm_config.VSC_A100_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
```

## Code Modifications

The following files were modified to support VSC:

1. **`code/src/optexp/runner/slurm/slurm_config.py`**
   - Added VSC A100 configurations

2. **`code/src/optexp/runner/slurm/sbatch_writers.py`**
   - Added support for `--cluster` and `--partition` SBATCH directives

3. **`code/src/optexp/config/__init__.py`**
   - Added `get_slurm_cluster()` and `get_slurm_partition()` functions
   - These read from `OPTEXP_SLURM_CLUSTER` and `OPTEXP_SLURM_PARTITION` env vars

## Important Notes

- **Storage**: Use `$VSC_DATA` for persistent storage (not `$VSC_SCRATCH`)
- **Credits**: Check your remaining credits with `mam-balance`
- **Job Status**: Monitor jobs with `squeue -u $USER`
- **VSC Account**: Format is usually `lp_projectname` or your research group account

## Getting Help

- VSC Documentation: https://docs.vscentrum.be/
- Check module availability: `module av Python` or `module av CUDA`
- VSC support: hpc@kuleuven.be

See [VSC_ONDEMAND_GUIDE.md](../VSC_ONDEMAND_GUIDE.md) for detailed instructions.
