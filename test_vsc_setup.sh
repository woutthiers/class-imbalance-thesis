#!/bin/bash
# Quick test script to verify VSC setup
# Run with: sbatch test_vsc_setup.sh

#SBATCH --account=lp_your_project     # CHANGE THIS
#SBATCH --time=0-00:10
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --cluster=wice
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@kuleuven.be  # CHANGE THIS

echo "==============================================="
echo "VSC Setup Test Started"
echo "==============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Load modules and activate environment
echo "Loading modules..."
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
echo "Modules loaded successfully"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source $VSC_DATA/venvs/optexp/bin/activate
echo "Virtual environment activated"
echo ""

# Test Python version
echo "Python version:"
python --version
echo ""

# Test PyTorch installation
echo "Testing PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Test optexp installation
echo "Testing optexp installation..."
python -c "import optexp; print('optexp imported successfully')"
echo ""

# Load environment variables
echo "Loading environment variables..."
source code/envs/env_vsc.sh
echo "Workspace: $OPTEXP_WORKSPACE"
echo ""

# Test workspace directory
echo "Testing workspace directory..."
ls -la $OPTEXP_WORKSPACE
echo ""

echo "==============================================="
echo "VSC Setup Test Completed Successfully!"
echo "==============================================="
