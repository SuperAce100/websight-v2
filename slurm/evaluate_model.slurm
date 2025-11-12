#!/bin/bash
#SBATCH --job-name=qwen3vl_eval
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/evaluate_qwen3vl_%j.out
#SBATCH --error=logs/evaluate_qwen3vl_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Qwen3-VL-8B Inference Evaluation on Test Set (NO GRADIENTS)
# Evaluation time: ~1-4 hours depending on test set size
#
# This script runs INFERENCE-ONLY evaluation with NO gradient computation.
# The input format matches training exactly (qwen2_vl template applied automatically).
#
# Usage:
#   # For merged model (uploaded to HuggingFace):
#   sbatch --export=MODEL_PATH="your-org/your-model-name" slurm/evaluate_model.slurm
#
#   # For LoRA adapter (local path):
#   sbatch --export=MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct",ADAPTER_PATH="saves/qwen3-vl-8b/lora/sft" slurm/evaluate_model.slurm
#
# The script will automatically:
#   1. Prepare test split from wave-ui/data.jsonl or base/prompts/prompts.jsonl (if needed)
#   2. Load the model from HuggingFace (or with LoRA adapter if specified)
#   3. Run inference on test set (no gradients, pure inference)
#   4. Save predictions to results/evaluation_results_<job_id>.jsonl
#
# Predictions (click locations) are saved in JSONL format with model outputs.
# Each line contains a JSON object with the model's prediction (e.g., "pyautogui.click(x, y)").

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Qwen3-VL Model Inference Evaluation Job"
echo "Mode: INFERENCE ONLY (No Gradients)"
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
mkdir -p results
mkdir -p .cache/huggingface

# Print system information
echo ""
echo "========================================"
echo "System Information"
echo "========================================"
nvidia-smi
echo ""
python --version
echo ""

# Check if test file exists, if not, prepare it
echo "========================================"
echo "Checking test dataset..."
echo "========================================"

if [ ! -f "data/wave_ui_test.jsonl" ]; then
    echo "Test file not found. Preparing test split..."
    python scripts/prepare_test_split.py \
        --output data/wave_ui_test.jsonl \
        --base-image-path wave-ui
    
    if [ ! -f "data/wave_ui_test.jsonl" ]; then
        echo "✗ Error: Failed to create test file"
        exit 1
    fi
else
    echo "✓ Test file found: data/wave_ui_test.jsonl"
fi

TEST_COUNT=$(wc -l < data/wave_ui_test.jsonl)
echo "✓ Test samples: ${TEST_COUNT}"
echo ""

# Verify model path (user should set this)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

echo "========================================"
echo "Model Configuration"
echo "========================================"
echo "Model: ${MODEL_PATH}"
if [ -n "${ADAPTER_PATH}" ]; then
    echo "Adapter: ${ADAPTER_PATH}"
fi
echo "Test file: data/wave_ui_test.jsonl"
echo "Media directory: wave-ui"
echo ""

# Run inference evaluation (no gradients)
echo "========================================"
echo "Starting inference evaluation..."
echo "Mode: INFERENCE ONLY (No Gradients)"
echo "Input format: Same as training (qwen2_vl template)"
echo "========================================"

# Build command
EVAL_CMD="python scripts/test_after_grounding.py \
    --model-name-or-path ${MODEL_PATH} \
    --test-file data/wave_ui_test.jsonl \
    --media-dir wave-ui \
    --output results/evaluation_results_${SLURM_JOB_ID}.jsonl \
    --device cuda"

if [ -n "${ADAPTER_PATH}" ]; then
    EVAL_CMD="${EVAL_CMD} --adapter-path ${ADAPTER_PATH}"
fi

# Run inference evaluation (no gradients, pure inference)
eval ${EVAL_CMD}

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Inference evaluation completed successfully!"
    echo ""
    echo "Predictions (click locations) saved to: results/evaluation_results_${SLURM_JOB_ID}.jsonl"
    echo ""
    echo "The predictions file contains model outputs in JSONL format."
    echo "Each line contains a JSON object with the model's prediction (e.g., 'pyautogui.click(x, y)')."
    echo ""
    # Check if results file exists and show sample
    if [ -f "results/evaluation_results_${SLURM_JOB_ID}.jsonl" ]; then
        echo "File info:"
        PRED_COUNT=$(wc -l < results/evaluation_results_${SLURM_JOB_ID}.jsonl)
        echo "  Total predictions: ${PRED_COUNT}"
        echo ""
        echo "Sample prediction (first line):"
        head -n 1 results/evaluation_results_${SLURM_JOB_ID}.jsonl | python -m json.tool 2>/dev/null || head -n 1 results/evaluation_results_${SLURM_JOB_ID}.jsonl
    fi
else
    echo "✗ Inference evaluation failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

