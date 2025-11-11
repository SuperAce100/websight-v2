#!/bin/bash
# Download and extract WebSight-v2 dataset
# Usage: ./scripts/download_dataset.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "WebSight-v2 Dataset Download"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Configuration
DEST_DIR="${DEST_DIR:-/hai/scratch/websight-v2/data}"
TEMP_DIR="${TEMP_DIR:-/tmp/websight-v2-download-$$}"

echo "Destination: ${DEST_DIR}"
echo "Temp directory: ${TEMP_DIR}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${WORKSPACE_DIR}"

echo "Workspace: ${WORKSPACE_DIR}"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Create directories
mkdir -p logs
mkdir -p "${TEMP_DIR}"

# Check if dataset already exists
if [ -f "${DEST_DIR}/prompts.jsonl" ]; then
    echo "⚠ Warning: Dataset already exists at ${DEST_DIR}"
    read -p "Overwrite? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo "Removing existing dataset..."
    rm -rf "${DEST_DIR}"
fi

# =============================================================================
# CONFIGURATION: Set your download source here
# =============================================================================

# Option 1: Download from HuggingFace (DEFAULT - agentsea/wave-ui)
# This downloads parquet files and extracts to JSONL + images automatically
HF_REPO="${HF_REPO:-agentsea/wave-ui}"

# Option 2: Download from URL (tar.gz archive)
# DOWNLOAD_URL="https://example.com/websight-v2-dataset.tar.gz"

# Option 3: Copy from local path
# LOCAL_PATH="/path/to/existing/dataset"

# =============================================================================

echo "========================================"
echo "Downloading dataset..."
echo "========================================"
echo ""

# Choose download method (default: HuggingFace)

if [ ! -z "${LOCAL_PATH:-}" ]; then
    echo "Method: Local copy"
    python scripts/download_dataset.py \
        --local-path "${LOCAL_PATH}" \
        --dest "${DEST_DIR}"

elif [ ! -z "${DOWNLOAD_URL:-}" ]; then
    echo "Method: Direct URL download"
    python scripts/download_dataset.py \
        --url "${DOWNLOAD_URL}" \
        --dest "${DEST_DIR}" \
        --temp-dir "${TEMP_DIR}"

else
    # Default to HuggingFace
    echo "Method: HuggingFace Hub (${HF_REPO})"
    echo "This will download parquet files and extract to JSONL + images"
    python scripts/download_dataset.py \
        --hf-repo "${HF_REPO}" \
        --dest "${DEST_DIR}"
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
    echo "  ./scripts/prepare_data.sh"
else
    echo "✗ Dataset download failed with exit code ${EXIT_CODE}"
    echo "Check the error messages above for details."
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

