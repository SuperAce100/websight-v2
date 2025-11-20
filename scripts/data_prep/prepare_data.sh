#!/bin/bash
# Prepare training data from WebSight-v2 dataset
# Usage: ./scripts/prepare_data.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Qwen3-VL Data Preparation"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${WORKSPACE_DIR}"

echo "Workspace: ${WORKSPACE_DIR}"
echo ""

# Dataset location
DATA_DIR="${DATA_DIR:-/hai/scratch/asanshay/websight-v2/data}"

# Verify dataset exists
if [ ! -f "${DATA_DIR}/prompts.jsonl" ]; then
    echo "✗ Error: Dataset not found at ${DATA_DIR}/prompts.jsonl"
    echo ""
    echo "Please download the dataset first:"
    echo "  ./scripts/download_dataset.sh"
    echo ""
    echo "Or set a custom location:"
    echo "  DATA_DIR=/path/to/data ./scripts/prepare_data.sh"
    exit 1
fi

echo "✓ Found dataset at ${DATA_DIR}/prompts.jsonl"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p logs

# Run data transformation
echo ""
echo "========================================"
echo "Transforming dataset..."
echo "========================================"
echo ""

python scripts/transform_for_training.py \
    --input "${DATA_DIR}/prompts.jsonl" \
    --output-dir data \
    --base-image-path "${DATA_DIR}" \
    --val-ratio 0.1 \
    --seed 42

EXIT_CODE=$?

# Verify output
echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Verifying output..."
    echo "========================================"
    
    if [ -f "data/wave_ui_train.jsonl" ]; then
        TRAIN_COUNT=$(wc -l < data/wave_ui_train.jsonl)
        echo "✓ Training set: ${TRAIN_COUNT} samples"
    else
        echo "✗ Error: Training set not created!"
        EXIT_CODE=1
    fi
    
    if [ -f "data/wave_ui_val.jsonl" ]; then
        VAL_COUNT=$(wc -l < data/wave_ui_val.jsonl)
        echo "✓ Validation set: ${VAL_COUNT} samples"
    else
        echo "✗ Error: Validation set not created!"
        EXIT_CODE=1
    fi
    
    # Show sample
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ""
        echo "Sample from training set:"
        head -n 1 data/wave_ui_train.jsonl | python -m json.tool || true
    fi
fi

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Data preparation complete!"
    echo ""
    echo "Next steps:"
    echo "  sbatch slurm/train_qwen_vl.slurm  # For GPU training"
    echo "  # or"
    echo "  python scripts/train.py --num-gpus 8  # Direct training"
else
    echo "✗ Data preparation failed!"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

