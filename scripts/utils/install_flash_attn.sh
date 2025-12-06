#!/bin/bash
# Install Flash Attention 2 after torch is already installed
# This script handles the common flash-attn installation issue

set +e  # Don't exit on error for this optional package

echo "Installing Flash Attention 2..."
echo "This requires torch to be installed first and may take 5-10 minutes..."
echo ""

# Check if torch is installed
python -c "import torch; print(f'✓ Found PyTorch {torch.__version__}')" || {
    echo "✗ PyTorch not found. Please install it first:"
    echo "  pip install torch>=2.1.0"
    exit 1
}

# Install flash-attn
echo ""
echo "Compiling Flash Attention from source..."
pip install flash-attn --no-build-isolation

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Flash Attention 2 installed successfully"
else
    echo ""
    echo "⚠ Flash Attention installation failed"
    echo ""
    echo "This is optional - training will still work without it,"
    echo "but may be slower and use more memory."
    echo ""
    echo "Common issues:"
    echo "  - Missing CUDA toolkit"
    echo "  - Incompatible GPU architecture"
    echo "  - Insufficient memory during compilation"
    echo ""
    echo "You can continue without it."
fi

