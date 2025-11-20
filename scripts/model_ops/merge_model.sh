#!/bin/bash
# Simple wrapper script for merge_model.py

set -e

# Default values
ADAPTER_PATH="${1:-/hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200}"
OUTPUT_DIR="${2:-/hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

# Check if adapter exists
if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "❌ Error: Adapter path not found: ${ADAPTER_PATH}"
    exit 1
fi

if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
    echo "❌ Error: adapter_config.json not found in ${ADAPTER_PATH}"
    exit 1
fi

echo "========================================"
echo "Merge LoRA Adapter with Base Model"
echo "========================================"
echo "Adapter: ${ADAPTER_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "Base:    ${BASE_MODEL}"
echo "========================================"
echo ""

# Run merge script
python scripts/merge_model.py \
    --adapter_path "${ADAPTER_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --base_model "${BASE_MODEL}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Merge completed!"
    echo "========================================"
    echo ""
    echo "To download to your local computer:"
    echo "  rsync -avz --progress ${OUTPUT_DIR}/ ~/local/path/qwen3-vl-8b-merged/"
    echo ""
    echo "To push to HuggingFace (after setting HF_TOKEN):"
    echo "  python scripts/merge_model.py \\"
    echo "    --adapter_path ${ADAPTER_PATH} \\"
    echo "    --output_dir ${OUTPUT_DIR} \\"
    echo "    --push_to_hub \\"
    echo "    --hub_model_id your-username/qwen3-vl-8b-websight \\"
    echo "    --hub_token \${HF_TOKEN}"
else
    echo ""
    echo "❌ Merge failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}

