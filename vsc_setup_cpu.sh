#!/bin/bash
# VSC wice cluster setup script - CPU ONLY (no CUDA)
# Run this script to set up your environment: source vsc_setup_cpu.sh

echo "Setting up CPU-only environment for VSC wice cluster..."

# Load required modules (no CUDA)
module purge
module load Python/3.10.4-GCCcore-11.3.0

# Create virtual environment if it doesn't exist
VENV_PATH="$VSC_DATA/venvs/optexp"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python -m venv "$VENV_PATH"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
echo "Virtual environment activated: $VENV_PATH"

# Verify Python version
python --version

echo ""
echo "Environment setup complete (CPU-only mode)!"
echo "To install dependencies (first time only), run:"
echo "  cd code"
echo "  pip install -r requirements/main.txt"
echo "  pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu"
echo "  pip install torchtext==0.14.1 portalocker==2.7.0 lightning==2.0.9 torchdata==0.5.1"
echo "  pip install -e ."
echo ""
echo "After installation, source the environment variables:"
echo "  source code/envs/env_vsc.sh"
echo ""
echo "NOTE: Using CPU-only PyTorch. Experiments will run slower but don't require GPU allocation."
