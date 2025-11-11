#!/bin/bash
# Quick setup script for Qwen3-VL fine-tuning environment

set -e

echo "========================================"
echo "Qwen3-VL Fine-tuning Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
else
    echo ""
    echo "Virtual environment already exists. Activating..."
    source venv/bin/activate
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements-training.txt

# Clone LLaMA-Factory if not present
if [ ! -d "LLaMA-Factory" ]; then
    echo ""
    echo "Cloning LLaMA-Factory..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e .
    cd ..
else
    echo ""
    echo "✓ LLaMA-Factory already cloned"
fi

# Install latest transformers for Qwen3-VL support
echo ""
echo "Installing latest Transformers from source..."
pip install git+https://github.com/huggingface/transformers

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p saves/qwen3-vl-8b/lora/sft
mkdir -p .cache/huggingface

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/transform_for_training.py
chmod +x scripts/train.py
chmod +x slurm/prepare_data.slurm
chmod +x slurm/train_qwen_vl.slurm

# Verify installation
echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"

python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "⚠ PyTorch not found"
python3 -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" 2>/dev/null || echo "⚠ Transformers not found"
python3 -c "import deepspeed; print(f'✓ DeepSpeed {deepspeed.__version__}')" 2>/dev/null || echo "⚠ DeepSpeed not found (optional)"
python3 -c "import peft; print(f'✓ PEFT {peft.__version__}')" 2>/dev/null || echo "⚠ PEFT not found"

# Check CUDA
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo "✓ CUDA available with ${GPU_COUNT} GPU(s)"
else
    echo "⚠ CUDA not available (required for training)"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Dataset location: /hai/scratch/websight-v2/data"
echo ""
echo "Next steps:"
echo "1. Verify dataset access:"
echo "   ls -lh /hai/scratch/websight-v2/data/prompts.jsonl"
echo ""
echo "2. Prepare the training data:"
echo "   python3 scripts/transform_for_training.py"
echo "   # Or: sbatch slurm/prepare_data.slurm"
echo ""
echo "3. (Optional) Test on single GPU:"
echo "   python3 scripts/train.py --num-gpus 1 --no-deepspeed"
echo ""
echo "4. Submit SLURM job for full training:"
echo "   sbatch slurm/train_qwen_vl.slurm"
echo ""
echo "See TRAINING_README.md and DATASET_INFO.md for details."
echo ""

