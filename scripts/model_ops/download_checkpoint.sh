#!/bin/bash
# Download checkpoint from cluster to local computer
# Usage: ./scripts/download_checkpoint.sh [destination]

CLUSTER_USER="${CLUSTER_USER:-asanshay}"
CLUSTER_HOST="${CLUSTER_HOST:-haic}"
REMOTE_PATH="/hai/scratch/asanshay/websight-v2/checkpoint-200.tar.gz"
LOCAL_DEST="${1:-~/Downloads/checkpoint-200.tar.gz}"

echo "Downloading checkpoint from cluster..."
echo "  From: ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_PATH}"
echo "  To:   ${LOCAL_DEST}"
echo ""

# Create destination directory if needed
mkdir -p "$(dirname "${LOCAL_DEST/#\~/$HOME}")"

# Download using rsync (resumable)
rsync -avz --partial --progress \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_PATH}" \
    "${LOCAL_DEST/#\~/$HOME}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Download complete!"
    echo "  File: ${LOCAL_DEST}"
    echo ""
    echo "To extract:"
    echo "  cd $(dirname "${LOCAL_DEST/#\~/$HOME}")"
    echo "  tar -xzf checkpoint-200.tar.gz"
else
    echo ""
    echo "❌ Download failed!"
    exit 1
fi

