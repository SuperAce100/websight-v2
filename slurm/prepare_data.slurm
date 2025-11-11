#!/bin/bash
#SBATCH --job-name=prepare_qwen_data
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Data preparation job for Qwen3-VL fine-tuning
# This job transforms the wave-ui dataset into the format required by LLaMA-Factory

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Qwen3-VL Data Preparation Job"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Load required modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/12.1

# Activate virtual environment
# Adjust this path to your actual virtual environment
WORKSPACE_DIR="/Users/asanshaygupta/Documents/Codes/websight-v2"
cd "${WORKSPACE_DIR}"

# If using uv (as per user preferences)
if command -v uv &> /dev/null; then
    echo "Using uv for Python environment management"
    # uv will handle the environment
else
    # Fallback to traditional venv
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Warning: No virtual environment found. Using system Python."
    fi
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

python scripts/transform_for_training.py \
    --input wave-ui/prompts.jsonl \
    --output-dir data \
    --base-image-path wave-ui \
    --val-ratio 0.1 \
    --seed 42

# Verify output
echo ""
echo "========================================"
echo "Verifying output..."
echo "========================================"

if [ -f "data/wave_ui_train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/wave_ui_train.jsonl)
    echo "✓ Training set: ${TRAIN_COUNT} samples"
else
    echo "✗ Error: Training set not created!"
    exit 1
fi

if [ -f "data/wave_ui_val.jsonl" ]; then
    VAL_COUNT=$(wc -l < data/wave_ui_val.jsonl)
    echo "✓ Validation set: ${VAL_COUNT} samples"
else
    echo "✗ Error: Validation set not created!"
    exit 1
fi

# Show sample
echo ""
echo "Sample from training set:"
head -n 1 data/wave_ui_train.jsonl | python -m json.tool

echo ""
echo "========================================"
echo "Data preparation complete!"
echo "End time: $(date)"
echo "========================================"

