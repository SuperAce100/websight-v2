#!/bin/bash
#SBATCH --job-name=merge_qwen3vl
#SBATCH --account=ingrai
#SBATCH --output=logs/merge_model_%j.out
#SBATCH --error=logs/merge_model_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Merge LoRA adapter with base model
# This script merges the LoRA adapter with the base model, creating a standalone
# merged model that can be downloaded to your local computer or pushed to HuggingFace.
#
# Usage:
#   # Just merge (for download)
#   sbatch --account=ingrai slurm/merge_model.slurm
#
#   # Merge and push to HuggingFace
#   export HF_TOKEN="hf_xxxxxxxxxxxxx"
#   export HF_USERNAME="your-username"
#   sbatch --account=ingrai slurm/merge_model.slurm

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Qwen3-VL Merge Model"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Configuration
WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

# Model paths - customize these if needed
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/qwen3-vl-8b-merged}"

# HuggingFace configuration (optional)
# Set these environment variables before running to push to HuggingFace:
# export HF_USERNAME="your-username"
# export HF_TOKEN="your-token"
# export HUB_MODEL_ID="your-username/qwen3-vl-8b-websight"  # Optional, defaults to username/qwen3-vl-8b-websight-merged
HF_USERNAME="${HF_USERNAME:-}"
HF_TOKEN="${HF_TOKEN:-}"
HUB_MODEL_ID="${HUB_MODEL_ID:-}"

# Options
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"  # Set to "true" to push to HuggingFace
PRIVATE_REPO="${PRIVATE_REPO:-false}"  # Set to "true" for private repository

echo "Configuration:"
echo "  Base model: ${BASE_MODEL}"
echo "  Adapter path: ${ADAPTER_PATH}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Push to HuggingFace: ${PUSH_TO_HUB}"
echo ""

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

# Set environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORKSPACE_DIR}/.cache/huggingface/datasets"
mkdir -p "${HF_HOME}"
mkdir -p "${WORKSPACE_DIR}/logs"

# Verify adapter exists
echo "========================================"
echo "Verifying LoRA adapter..."
echo "========================================"

if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "❌ Error: Adapter path not found: ${ADAPTER_PATH}"
    echo "Please set ADAPTER_PATH environment variable or edit this script."
    exit 1
fi

if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
    echo "❌ Error: adapter_config.json not found in ${ADAPTER_PATH}"
    exit 1
fi

echo "✓ LoRA adapter found"
echo ""

# Check if we need to push to HuggingFace
if [ "${PUSH_TO_HUB}" = "true" ]; then
    if [ -z "${HF_USERNAME}" ]; then
        echo "❌ Error: HF_USERNAME not set!"
        echo "Please set your HuggingFace username:"
        echo "  export HF_USERNAME='your-username'"
        exit 1
    fi
    
    if [ -z "${HF_TOKEN}" ]; then
        echo "❌ Error: HF_TOKEN not set!"
        echo "Please set your HuggingFace token:"
        echo "  export HF_TOKEN='hf_xxxxxxxxxxxxx'"
        echo ""
        echo "Get your token from: https://huggingface.co/settings/tokens"
        exit 1
    fi
    
    # Set default hub model ID if not provided
    if [ -z "${HUB_MODEL_ID}" ]; then
        HUB_MODEL_ID="${HF_USERNAME}/qwen3-vl-8b-websight-merged"
    fi
    
    echo "HuggingFace configuration:"
    echo "  Username: ${HF_USERNAME}"
    echo "  Repository: ${HUB_MODEL_ID}"
    echo "  Private: ${PRIVATE_REPO}"
    echo ""
fi

# Run merge script
echo "========================================"
echo "Merging LoRA adapter with base model..."
echo "========================================"
echo "This may take 10-30 minutes depending on model size..."
echo ""

# Build command
MERGE_CMD="python scripts/merge_model.py \
    --adapter_path \"${ADAPTER_PATH}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --base_model \"${BASE_MODEL}\""

# Add HuggingFace options if pushing
if [ "${PUSH_TO_HUB}" = "true" ]; then
    MERGE_CMD="${MERGE_CMD} \
        --push_to_hub \
        --hub_model_id \"${HUB_MODEL_ID}\" \
        --hub_token \"${HF_TOKEN}\""
    
    if [ "${PRIVATE_REPO}" = "true" ]; then
        MERGE_CMD="${MERGE_CMD} --private"
    fi
fi

# Execute merge
echo "Running merge command..."
eval ${MERGE_CMD}

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Merge completed successfully!"
    echo "========================================"
    echo ""
    echo "Merged model saved to: ${OUTPUT_DIR}"
    echo ""
    
    # Check model size
    if [ -d "${OUTPUT_DIR}" ]; then
        MODEL_SIZE=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
        echo "Model size: ${MODEL_SIZE}"
        echo ""
    fi
    
    echo "Next steps:"
    echo ""
    echo "1. Download to your local computer:"
    echo "   rsync -avz --progress \\"
    echo "     ${USER}@haic:${WORKSPACE_DIR}/${OUTPUT_DIR}/ \\"
    echo "     ~/local/path/qwen3-vl-8b-merged/"
    echo ""
    
    if [ "${PUSH_TO_HUB}" = "true" ]; then
        echo "2. Model is available on HuggingFace:"
        echo "   https://huggingface.co/${HUB_MODEL_ID}"
        echo ""
    else
        echo "2. To push to HuggingFace later, run:"
        echo "   export HF_TOKEN=\"hf_xxxxxxxxxxxxx\""
        echo "   export HF_USERNAME=\"your-username\""
        echo "   python scripts/merge_model.py \\"
        echo "     --adapter_path ${ADAPTER_PATH} \\"
        echo "     --output_dir ${OUTPUT_DIR} \\"
        echo "     --push_to_hub \\"
        echo "     --hub_model_id your-username/qwen3-vl-8b-websight \\"
        echo "     --hub_token \$HF_TOKEN"
        echo ""
    fi
    
    echo "3. Use for inference:"
    echo "   python test_model.py \\"
    echo "     --model-path ${OUTPUT_DIR} \\"
    echo "     --image path/to/image.png \\"
    echo "     --prompt \"click the button\""
else
    echo "✗ Merge failed with exit code ${EXIT_CODE}"
    echo "========================================"
    echo ""
    echo "Check the error log for details:"
    echo "  logs/merge_model_${SLURM_JOB_ID}.err"
fi

echo ""
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

