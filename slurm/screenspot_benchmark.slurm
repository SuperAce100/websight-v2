#!/bin/bash
#SBATCH --job-name=screenspot_benchmark
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/screenspot_benchmark_%j.out
#SBATCH --error=logs/screenspot_benchmark_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ScreenSpot-Pro Benchmark Job
# Runs end-to-end benchmark: dataset preparation, inference, and evaluation
#
# This script:
# 1. Prepares ScreenSpot-Pro dataset (if not already cached)
# 2. Runs inference using specified model
# 3. Evaluates predictions and generates metrics
#
# Usage:
#   # Default: Use Asanshay/websight-v2-grounded model
#   sbatch slurm/screenspot_benchmark.slurm
#
#   # Custom model
#   sbatch --export=MODEL_PATH="your-org/your-model" slurm/screenspot_benchmark.slurm
#
#   # With LoRA adapter
#   sbatch --export=MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct",ADAPTER_PATH="ckpts/checkpoint-200" slurm/screenspot_benchmark.slurm
#
#   # Test on subset
#   sbatch --export=LIMIT=100 slurm/screenspot_benchmark.slurm

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "ScreenSpot-Pro Benchmark Job"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Start time: $(date)"
echo ""

# Set workspace directory
WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORKSPACE_DIR}/.cache/huggingface/datasets"

# Create necessary directories
mkdir -p logs
mkdir -p runs/screenspot_pro
mkdir -p .cache/huggingface

# Configuration from environment or defaults
MODEL_PATH="${MODEL_PATH:-Asanshay/websight-v2-grounded}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATA_DIR="${DATA_DIR:-screenspot_pro}"
IMAGES_DIR="${IMAGES_DIR:-/hai/scratch/asanshay/screenspot-pro/images}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# Output files
PREDICTIONS_FILE="runs/screenspot_pro/predictions_${SLURM_JOB_ID}.jsonl"
RESULTS_JSON="runs/screenspot_pro/results_${SLURM_JOB_ID}.json"

# Print system information
echo ""
echo "========================================"
echo "System Information"
echo "========================================"
nvidia-smi
echo ""
python --version
echo ""

# Print configuration
echo "========================================"
echo "Configuration"
echo "========================================"
echo "Model: ${MODEL_PATH}"
if [ -n "${ADAPTER_PATH}" ]; then
    echo "Adapter: ${ADAPTER_PATH}"
fi
echo "Data directory: ${DATA_DIR}"
echo "Images directory: ${IMAGES_DIR}"
echo "Predictions: ${PREDICTIONS_FILE}"
echo "Results: ${RESULTS_JSON}"
if [ -n "${LIMIT}" ]; then
    echo "Limit: ${LIMIT} samples"
fi
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo ""

# =============================================================================
# Step 1: Prepare Dataset
# =============================================================================

echo "========================================"
echo "Step 1: Prepare ScreenSpot-Pro Dataset"
echo "========================================"
echo ""

DATA_JSONL="${DATA_DIR}/data.jsonl"

# Check if dataset is complete (both data.jsonl and images directory)
if [ -f "${DATA_JSONL}" ] && [ -d "${IMAGES_DIR}" ]; then
    SAMPLE_COUNT=$(wc -l < "${DATA_JSONL}")
    IMAGE_COUNT=$(find "${IMAGES_DIR}" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    
    echo "✓ Dataset already exists at ${DATA_JSONL}"
    echo "  Samples: ${SAMPLE_COUNT}"
    echo "  Images: ${IMAGE_COUNT}"
    
    # Verify image count matches sample count (roughly)
    if [ ${IMAGE_COUNT} -lt $((SAMPLE_COUNT - 10)) ]; then
        echo "⚠️  Warning: Image count (${IMAGE_COUNT}) is significantly less than sample count (${SAMPLE_COUNT})"
        echo "  Re-downloading dataset to ensure completeness..."
        NEED_DOWNLOAD=1
    else
        NEED_DOWNLOAD=0
    fi
else
    echo "Dataset not found or incomplete"
    NEED_DOWNLOAD=1
fi

if [ ${NEED_DOWNLOAD} -eq 1 ]; then
    echo "Downloading and preparing dataset..."
    python scripts/prepare_screenspot_pro.py \
        --output-dir "${DATA_DIR}" \
        --images-dir "${IMAGES_DIR}" \
        --max-retries 10 \
        --retry-delay 120
    
    if [ ! -f "${DATA_JSONL}" ]; then
        echo "✗ Error: Failed to prepare dataset"
        exit 1
    fi
    
    if [ ! -d "${IMAGES_DIR}" ]; then
        echo "✗ Error: Images directory not created"
        exit 1
    fi
    
    SAMPLE_COUNT=$(wc -l < "${DATA_JSONL}")
    IMAGE_COUNT=$(find "${IMAGES_DIR}" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    echo "✓ Dataset prepared: ${SAMPLE_COUNT} samples, ${IMAGE_COUNT} images"
fi

echo ""

# =============================================================================
# Step 2: Run Inference
# =============================================================================

echo "========================================"
echo "Step 2: Run Inference"
echo "========================================"
echo ""

# Build inference command
INFERENCE_CMD="python scripts/run_screenspot_benchmark.py \
    --model-name-or-path ${MODEL_PATH} \
    --data-dir ${DATA_DIR} \
    --images-dir ${IMAGES_DIR} \
    --output ${PREDICTIONS_FILE} \
    --device cuda \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --progress-interval 50"

if [ -n "${ADAPTER_PATH}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --adapter-path ${ADAPTER_PATH}"
fi

if [ -n "${LIMIT}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --limit ${LIMIT}"
fi

# Run inference
eval ${INFERENCE_CMD}

INFERENCE_EXIT=$?

if [ ${INFERENCE_EXIT} -ne 0 ]; then
    echo "✗ Error: Inference failed with exit code ${INFERENCE_EXIT}"
    exit ${INFERENCE_EXIT}
fi

# Check predictions file
if [ ! -f "${PREDICTIONS_FILE}" ]; then
    echo "✗ Error: Predictions file not created"
    exit 1
fi

PRED_COUNT=$(wc -l < "${PREDICTIONS_FILE}")
echo "✓ Inference complete: ${PRED_COUNT} predictions"
echo ""

# =============================================================================
# Step 3: Evaluate Predictions
# =============================================================================

echo "========================================"
echo "Step 3: Evaluate Predictions"
echo "========================================"
echo ""

python scripts/evaluate_screenspot.py \
    --predictions "${PREDICTIONS_FILE}" \
    --ground-truth "${DATA_JSONL}" \
    --output-json "${RESULTS_JSON}"

EVAL_EXIT=$?

if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "✗ Error: Evaluation failed with exit code ${EVAL_EXIT}"
    exit ${EVAL_EXIT}
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "========================================"
echo "✓ Benchmark Complete!"
echo "========================================"
echo ""
echo "Files:"
echo "  Predictions: ${PREDICTIONS_FILE}"
echo "  Results:     ${RESULTS_JSON}"
echo ""

if [ -f "${RESULTS_JSON}" ]; then
    echo "Summary:"
    # Extract key metrics from JSON
    python -c "
import json
with open('${RESULTS_JSON}', 'r') as f:
    results = json.load(f)
    print(f\"  Total samples:  {results['total']}\")
    print(f\"  Parsed:         {results['parsed']} ({results['parsed']/results['total']*100:.1f}%)\")
    print(f\"  Accuracy:       {results['correct']}/{results['parsed']} ({results['accuracy']:.2f}%)\")
    print(f\"  Avg distance:   {results['avg_distance']:.1f} pixels\")
" 2>/dev/null || echo "  (Could not parse results JSON)"
fi

echo ""
echo "End time: $(date)"
echo "========================================"

exit 0

