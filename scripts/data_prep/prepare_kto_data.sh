#!/bin/bash
# Prepare AgentNet data for KTO (Kahneman-Tversky Optimization) training
# - Transforms agentnet trajectories into KTO format with binary labels
# - Outputs combined file with positive/negative examples based on task_completed status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Default paths
DEFAULT_RAW_ROOT="/hai/scratch/asanshay/websight-v2/agentnet"
RAW_ROOT="${AGENTNET_DATA_ROOT:-${DEFAULT_RAW_ROOT}}"
MERGED_JSONL="${RAW_ROOT}/agentnet_all.jsonl"

DEFAULT_OUTPUT_DIR="${REPO_ROOT}/data"
OUTPUT_DIR="${AGENTNET_KTO_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
mkdir -p "${OUTPUT_DIR}"

DEFAULT_IMAGES_DIR="${RAW_ROOT}/images"
IMAGES_DIR="${AGENTNET_IMAGE_DIR:-${DEFAULT_IMAGES_DIR}}"

# KTO transformation parameters
MIN_TRAJ="${AGENTNET_KTO_MIN_TRAJ:-1}"
INCLUDE_REFLECTION="${AGENTNET_KTO_INCLUDE_REFLECTION:-0}"

echo "========================================"
echo "AgentNet KTO Data Preparation"
echo "========================================"
echo "Input     : ${MERGED_JSONL}"
echo "Output    : ${OUTPUT_DIR}/agentnet_kto.jsonl"
echo "Images    : ${IMAGES_DIR}"
echo ""

# Check input file exists
if [ ! -f "${MERGED_JSONL}" ]; then
    echo "✗ Error: ${MERGED_JSONL} not found."
    echo "  Run: sbatch slurm/prepare_agentnet.slurm first to create merged file"
    exit 1
fi

# Check images directory exists
if [ ! -d "${IMAGES_DIR}" ]; then
    echo "✗ Error: Images directory ${IMAGES_DIR} not found."
    exit 1
fi

# Determine Python command
TRANSFORM_CMD=(python)
if command -v uv >/dev/null 2>&1; then
    TRANSFORM_CMD=(uv run python)
fi

# Build transformation command
ARGS=(
    scripts/agentnet_scripts/transform_agentnet_for_kto.py
    --input "${MERGED_JSONL}"
    --output "${OUTPUT_DIR}/agentnet_kto.jsonl"
    --base-image-dir "${IMAGES_DIR}"
    --min-traj-length "${MIN_TRAJ}"
)

if [ "${INCLUDE_REFLECTION}" = "1" ]; then
    ARGS+=(--include-reflection)
fi

# Note: NOT using --split-by-label to get combined format with label field

echo "Running KTO transformation:"
printf '  %q' "${TRANSFORM_CMD[@]}" "${ARGS[@]}"
echo ""
echo ""

"${TRANSFORM_CMD[@]}" "${ARGS[@]}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "✗ Error: KTO transformation failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

# Verify output
OUTPUT_FILE="${OUTPUT_DIR}/agentnet_kto.jsonl"
if [ ! -f "${OUTPUT_FILE}" ]; then
    echo "✗ Error: Output file ${OUTPUT_FILE} not found."
    exit 1
fi

COUNT=$(wc -l < "${OUTPUT_FILE}")
echo ""
echo "========================================"
echo "✓ KTO data preparation complete!"
echo "========================================"
echo "Output file: ${OUTPUT_FILE}"
echo "Total examples: ${COUNT}"
echo ""
echo "Sample record:"
python3 - <<'PY'
import json
from pathlib import Path

sample_path = Path("data/agentnet_kto.jsonl")
if sample_path.exists():
    with sample_path.open("r", encoding="utf-8") as fh:
        line = fh.readline().strip()
        if line:
            obj = json.loads(line)
            preview = {
                "has_messages": bool(obj.get("messages")),
                "has_images": bool(obj.get("images")),
                "label": obj.get("label"),
                "message_count": len(obj.get("messages", [])),
            }
            print(json.dumps(preview, indent=2, ensure_ascii=False))
        else:
            print("{}")
PY

echo ""
echo "Next steps:"
echo "  1. Verify dataset_info.json includes agentnet_kto_train entry"
echo "  2. Run KTO training: sbatch slurm/train_agentnet_kto.slurm"
echo ""

