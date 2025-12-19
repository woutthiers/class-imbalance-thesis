# Using VSC OnDemand with VS Code

This guide explains how to set up and run experiments on the VSC wice cluster using OnDemand's VS Code integration.

## Step 1: Access VSC OnDemand

1. Go to https://login.hpc.kuleuven.be/
2. Log in with your VSC credentials
3. Navigate to **Interactive Apps** → **VS Code**

## Step 2: Request a VS Code Session

Configure your session:
- **Cluster**: wice
- **Partition**: interactive (for development) or gpu (for GPU jobs)
- **Number of hours**: 2-4 hours for development
- **Number of cores**: 2-4
- **Memory (GB)**: 8-16
- **Number of GPUs**: 0 for setup, 1 if you want to test with GPU

Click **Launch** and wait for the session to start.

## Step 3: Clone the Repository

Once VS Code opens in your browser:

1. Open the terminal in VS Code (Terminal → New Terminal)
2. Clone the repository:
   ```bash
   cd $VSC_DATA
   git clone https://github.com/fKunstner/class-imbalance-sgd-adam.git
   cd class-imbalance-sgd-adam
   ```

3. Open the folder in VS Code:
   - File → Open Folder
   - Navigate to `$VSC_DATA/class-imbalance-sgd-adam`

## Step 4: Set Up the Environment

In the VS Code terminal:

```bash
# Run the setup script
source vsc_setup.sh

# This will:
# - Load required modules (Python, CUDA, PyTorch)
# - Create a virtual environment at $VSC_DATA/venvs/optexp
# - Activate the virtual environment
```

## Step 5: Install Dependencies (First Time Only)

```bash
cd code

# Install main dependencies
pip install -r requirements/main.txt

# Install PyTorch with CUDA support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install additional dependencies
pip install torchtext==0.14.1 portalocker==2.7.0 lightning==2.0.9 torchdata==0.5.1

# Install the package
pip install -e .
```

## Step 6: Configure Environment Variables

Edit the file `code/envs/env_vsc.sh` and update:

```bash
export OPTEXP_WANDB_ENTITY=your_wandb_username      # Your W&B username
export WANDB_API_KEY=your_wandb_api_key_here        # Your W&B API key
export OPTEXP_SLURM_NOTIFICATION_EMAIL=your.email@kuleuven.be
export OPTEXP_SLURM_ACCOUNT=lp_your_project         # Your VSC project (e.g., lp_my_research)
```

Then load the environment:
```bash
source code/envs/env_vsc.sh
```

## Step 7: Verify Installation

```bash
# Test Python imports
python -c "import optexp; print('optexp imported successfully')"

# Test CUDA (if you requested a GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 8: Run Experiments

### Option A: Test Locally in VS Code Session

For quick testing in your interactive session:

```bash
cd code/src/optexp/experiments/toy_models

# Run a single experiment locally
python balanced_x_perclass.py --single 0

# Run all experiments locally (small experiments only!)
python balanced_x_perclass.py --local
```

### Option B: Submit to SLURM Queue

For longer experiments, submit to the cluster:

```bash
# Make sure environment is loaded
source ~/class-imbalance-sgd-adam/vsc_setup.sh
source code/envs/env_vsc.sh

cd code/src/optexp/experiments/toy_models

# Submit all experiments to SLURM
python balanced_x_perclass.py --slurm

# Check job status
squeue -u $USER

# View output
cat slurm-*.out
```

### Option C: Create Your Own Experiment

Create a new file in `code/src/optexp/experiments/toy_models/my_experiment.py`:

```python
from optexp import Experiment, exp_runner_cli
from optexp.datasets.classification_mixture import ClassificationMixture
from optexp.models.linear import LinearInit0
from optexp.optimizers import SGD_NM, Adam_NM
from optexp.problems.classification import FullBatchClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config
from optexp.utils import lr_grid

# Dataset: 100 classes with imbalance
dataset = ClassificationMixture(batch_size=1000, alpha=1.0, c=100, min_n=10)

# Linear classifier
model = LinearInit0(input=100, output=100, bias=True)

# Problem
problem = FullBatchClassificationWithPerClassStats(model, dataset)

# Optimizers
optimizers = [
    SGD_NM(lr) for lr in lr_grid(start=-3, end=-1, density=1)
] + [
    Adam_NM(lr) for lr in lr_grid(start=-5, end=-3, density=1)
]

# Experiments
experiments = [
    Experiment(optim=opt, problem=problem, group="MyLinearExperiment", 
               seed=0, epochs=100)
    for opt in optimizers
]

# Use VSC-specific SLURM config
SLURM_CONFIG = slurm_config.VSC_A100_1H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
```

Run it:
```bash
# Test one locally
python my_experiment.py --single 0

# Submit to cluster
python my_experiment.py --slurm
```

## Step 9: Download and Plot Results

After experiments complete:

```bash
# Download results from W&B
python balanced_x_perclass.py --download

# Generate plots
python balanced_x_perclass.py --plot

# Generate per-class plots
python balanced_x_perclass.py --plot --perclass
```

Plots are saved to `$VSC_DATA/optexp_workspace/plots/`

## Tips for OnDemand VS Code

1. **Use Git for Changes**: Commit and push changes regularly
2. **Monitor Resources**: Use `htop` to check CPU/memory usage
3. **Check Job Status**: Use `squeue -u $USER` to monitor SLURM jobs
4. **View Logs**: Check `slurm-*.out` files for job outputs
5. **Session Timeout**: Save work frequently; sessions expire after the requested time
6. **Multiple Terminals**: Use VS Code's split terminal for convenience

## Common Issues

**"Module not found" errors:**
```bash
# Make sure you've loaded the environment
source ~/class-imbalance-sgd-adam/vsc_setup.sh
source code/envs/env_vsc.sh
```

**SLURM job fails:**
- Check `slurm-*.out` for error messages
- Verify your VSC account is active: `mam-balance`
- Ensure you have sufficient credits

**Git authentication:**
```bash
# Use HTTPS with personal access token or set up SSH keys
git config --global credential.helper store
```

## Next Steps

- Explore other experiment files in `code/src/optexp/experiments/`
- Modify existing experiments or create new ones
- Check VSC documentation: https://docs.vscentrum.be/
