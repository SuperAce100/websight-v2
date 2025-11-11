#!/bin/bash
# Complete setup script for Qwen3-VL fine-tuning
# Creates venv, installs dependencies, and prepares environment
# Usage: ./setup.sh

set -e  # Exit on error

echo "========================================"
echo "Qwen3-VL Fine-tuning Setup"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Working directory: ${SCRIPT_DIR}"
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python version: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found!"
    exit 1
fi

# Create virtual environment
if [ -d "venv" ]; then
    echo ""
    echo "Virtual environment already exists."
    read -p "Recreate it? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf venv
    else
        echo "Using existing venv."
    fi
fi

if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated: $(which python)"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
echo ""

pip install -r requirements-training.txt

# Install transformers from source for latest Qwen3-VL support
echo ""
echo "Installing latest Transformers from source..."
pip install git+https://github.com/huggingface/transformers --quiet

# Clone LLaMA-Factory if not present
if [ ! -d "LLaMA-Factory" ]; then
    echo ""
    echo "Cloning LLaMA-Factory..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e . --quiet
    cd ..
    echo "✓ LLaMA-Factory installed"
else
    echo ""
    echo "✓ LLaMA-Factory already exists"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p configs
mkdir -p saves/qwen3-vl-8b/lora/sft
mkdir -p .cache/huggingface
echo "✓ Directories created"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true
chmod +x slurm/*.slurm 2>/dev/null || true
chmod +x setup_training.sh 2>/dev/null || true
echo "✓ Scripts are executable"

# Verify installation
echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"

python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch not found"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" || echo "✗ Transformers not found"
python -c "import peft; print(f'✓ PEFT {peft.__version__}')" || echo "✗ PEFT not found"
python -c "import deepspeed; print(f'✓ DeepSpeed {deepspeed.__version__}')" || echo "⚠ DeepSpeed not found (optional)"
python -c "import pyarrow; print(f'✓ PyArrow {pyarrow.__version__}')" || echo "✗ PyArrow not found"

# Check CUDA
echo ""
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "✓ CUDA available with ${GPU_COUNT} GPU(s)"
else
    echo "⚠ CUDA not available (required for training, but not for data prep)"
fi

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "Virtual environment: ${SCRIPT_DIR}/venv"
echo ""
echo "To activate the environment later:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo ""
echo "1. Download dataset:"
echo "   ./scripts/download_dataset.sh"
echo ""
echo "2. Prepare training data:"
echo "   ./scripts/prepare_data.sh"
echo ""
echo "3. Start training:"
echo "   sbatch slurm/train_qwen_vl.slurm"
echo ""
echo "See QUICKSTART.md for detailed instructions."
echo ""

