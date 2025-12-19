#!/bin/bash
# VSC wice cluster setup script
# Run this script to set up your environment: source vsc_setup.sh

echo "Setting up environment for VSC wice cluster..."

# Load required modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load PyTorch/1.13.1-foss-2022a-CUDA-11.7.0 2>/dev/null || echo "PyTorch module not available, will install via pip"

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
echo "Environment setup complete!"
echo "To install dependencies (first time only), run:"
echo "  cd code"
echo "  pip install -r requirements/main.txt"
echo "  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117"
echo "  pip install torchtext==0.14.1 portalocker==2.7.0 lightning==2.0.9 torchdata==0.5.1"
echo "  pip install -e ."
echo ""
echo "After installation, source the environment variables:"
echo "  source code/envs/env_vsc.sh"
