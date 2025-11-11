#!/bin/bash
#SBATCH --job-name=download_websight_data
#SBATCH --output=logs/download_dataset_%j.out
#SBATCH --error=logs/download_dataset_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Dataset download job for WebSight-v2
# Downloads and extracts the dataset to /hai/scratch/websight-v2/data
#
# Usage:
#   sbatch slurm/download_dataset.slurm

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "WebSight-v2 Dataset Download Job"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Load required modules (adjust based on your cluster)
# module load python/3.10

# Workspace and destination
WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

DEST_DIR="/hai/scratch/asanshay/websight-v2/data"
TEMP_DIR="${TMPDIR:-/tmp}/websight-v2-download-${SLURM_JOB_ID}"

echo "Workspace: ${WORKSPACE_DIR}"
echo "Destination: ${DEST_DIR}"
echo "Temp directory: ${TEMP_DIR}"
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

# Create necessary directories
mkdir -p logs
mkdir -p "${TEMP_DIR}"

# Ensure destination parent directory exists and is writable
DEST_PARENT=$(dirname "${DEST_DIR}")
if [ ! -d "${DEST_PARENT}" ]; then
    echo "Creating parent directory: ${DEST_PARENT}"
    mkdir -p "${DEST_PARENT}" || {
        echo "✗ Error: Cannot create parent directory ${DEST_PARENT}"
        echo "  You may need to run this manually or request access."
        exit 1
    }
fi

# Check if dataset already exists
if [ -f "${DEST_DIR}/prompts.jsonl" ]; then
    echo "⚠ Warning: Dataset already exists at ${DEST_DIR}"
    echo "Skipping download. To re-download, remove the existing dataset first:"
    echo "  rm -rf ${DEST_DIR}"
    exit 0
fi

# =============================================================================
# CONFIGURATION: Update these variables based on your dataset source
# =============================================================================

# Option 1: Download from URL
# DOWNLOAD_URL="https://example.com/websight-v2-dataset.tar.gz"

# Option 2: Download from HuggingFace
# HF_REPO="username/websight-v2"
# HF_FILE="dataset.tar.gz"

# Option 3: Copy from local path
# LOCAL_PATH="/path/to/existing/dataset"

# =============================================================================

echo ""
echo "========================================"
echo "Downloading dataset..."
echo "========================================"
echo ""

# Choose one of the following methods:

# Method 1: Direct URL download
if [ ! -z "${DOWNLOAD_URL:-}" ]; then
    echo "Method: Direct URL download"
    python scripts/download_dataset.py \
        --url "${DOWNLOAD_URL}" \
        --dest "${DEST_DIR}" \
        --temp-dir "${TEMP_DIR}"

# Method 2: HuggingFace download
elif [ ! -z "${HF_REPO:-}" ]; then
    echo "Method: HuggingFace Hub download"
    python scripts/download_dataset.py \
        --hf-repo "${HF_REPO}" \
        --hf-file "${HF_FILE}" \
        --dest "${DEST_DIR}" \
        --temp-dir "${TEMP_DIR}"

# Method 3: Local copy
elif [ ! -z "${LOCAL_PATH:-}" ]; then
    echo "Method: Local copy"
    python scripts/download_dataset.py \
        --local-path "${LOCAL_PATH}" \
        --dest "${DEST_DIR}"

else
    echo "✗ Error: No download method configured!"
    echo ""
    echo "Please edit slurm/download_dataset.slurm and set one of:"
    echo "  - DOWNLOAD_URL (for direct download)"
    echo "  - HF_REPO and HF_FILE (for HuggingFace)"
    echo "  - LOCAL_PATH (to copy from local directory)"
    echo ""
    exit 1
fi

EXIT_CODE=$?

# Clean up temp directory
if [ -d "${TEMP_DIR}" ]; then
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}"
fi

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Dataset download complete!"
    echo ""
    echo "Dataset location: ${DEST_DIR}"
    echo ""
    echo "Verification:"
    if [ -f "${DEST_DIR}/prompts.jsonl" ]; then
        RECORD_COUNT=$(wc -l < "${DEST_DIR}/prompts.jsonl")
        echo "  ✓ prompts.jsonl: ${RECORD_COUNT} records"
    fi
    if [ -d "${DEST_DIR}/images" ]; then
        IMAGE_COUNT=$(ls "${DEST_DIR}/images" | wc -l)
        echo "  ✓ images/: ${IMAGE_COUNT} files"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Prepare training data:"
    echo "     sbatch slurm/prepare_data.slurm"
    echo "  2. Start training:"
    echo "     sbatch slurm/train_qwen_vl.slurm"
else
    echo "✗ Dataset download failed with exit code ${EXIT_CODE}"
    echo "Check the error messages above for details."
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

