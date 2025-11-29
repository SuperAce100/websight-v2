#!/bin/bash
#SBATCH --job-name=prep_agentnet
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/prepare_agentnet_%j.out
#SBATCH --error=logs/prepare_agentnet_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# AgentNet → UI-TARS data preparation
# - Merges Windows/macOS + Ubuntu trajectories
# - Verifies extracted screenshots
# - Runs scripts/agentnet_scripts/transform_agentnet_for_training.py

set -euo pipefail

echo "========================================"
echo "AgentNet Data Preparation Job"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

mkdir -p logs

DEFAULT_SCRATCH_ROOT="/hai/scratch/asanshay/websight-v2/agentnet"
RAW_ROOT="${AGENTNET_DATA_ROOT:-${DEFAULT_SCRATCH_ROOT}}"
OUTPUT_DIR="${AGENTNET_OUTPUT_DIR:-${WORKSPACE_DIR}/data}"
IMAGES_DIR="${AGENTNET_IMAGE_DIR:-${RAW_ROOT}/images}"
MERGED_JSONL="${RAW_ROOT}/agentnet_all.jsonl"

VAL_RATIO="${AGENTNET_VAL_RATIO:-0.1}"
TEST_RATIO="${AGENTNET_TEST_RATIO:-0.05}"
MIN_TRAJ="${AGENTNET_MIN_TRAJ:-3}"
SEED="${AGENTNET_SEED:-42}"
MAX_EXAMPLES="${AGENTNET_MAX_EXAMPLES_PER_TRAJ:-}"
AUTO_DOWNLOAD="${AGENTNET_AUTO_DOWNLOAD:-1}"

echo "Workspace: ${WORKSPACE_DIR}"
echo "Raw dir : ${RAW_ROOT}"
echo "Images  : ${IMAGES_DIR}"
echo "Output  : ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Validate prerequisites
# ---------------------------------------------------------------------------

if [ ! -d "${RAW_ROOT}" ]; then
    if [ "${AUTO_DOWNLOAD}" != "1" ]; then
        cat <<EOF
✗ Error: ${RAW_ROOT} not found.
Either download xlangai/AgentNet manually or set AGENTNET_AUTO_DOWNLOAD=1 before submitting.
EOF
        exit 1
    fi

    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "✗ Error: huggingface-cli not found in PATH (required for auto-download)."
        exit 1
    fi

    echo "Downloading AgentNet dataset to ${RAW_ROOT} ..."
    mkdir -p "${RAW_ROOT}"

    HF_ARGS=()
    if [ -n "${HF_TOKEN:-}" ]; then
        HF_ARGS+=(--token "${HF_TOKEN}")
    fi

    huggingface-cli download xlangai/AgentNet \
        --repo-type dataset \
        --local-dir "${RAW_ROOT}" \
        --local-dir-use-symlinks False \
        "${HF_ARGS[@]}"
fi

mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${IMAGES_DIR}" ]; then
    cat <<EOF
✗ Error: Screenshot directory not found at ${IMAGES_DIR}

Expected layout:
  data/AgentNet/
    ├── agentnet_win_mac_18k.jsonl
    ├── agentnet_ubuntu_5k.jsonl
    └── images/
        ├── ea83c4aa-a4b1-48af-b439-0de7ee7b8d3f.png
        ├── ...

Please unzip the split archives first (per AgentNet README):
  cd data/AgentNet/win_mac_images
  zip -s 0 images.zip --out images_win_mac_full.zip
  unzip -n images_win_mac_full.zip -d ../images

  cd data/AgentNet/ubuntu_images
  zip -s 0 images.zip --out images_ubuntu_full.zip
  unzip -n images_ubuntu_full.zip -d ../images

Then ensure every *.png lives under data/AgentNet/images/.
EOF
    exit 1
fi

RAW_FILES=()
if [ -f "${RAW_ROOT}/agentnet_win_mac_18k.jsonl" ]; then
    RAW_FILES+=("${RAW_ROOT}/agentnet_win_mac_18k.jsonl")
fi
if [ -f "${RAW_ROOT}/agentnet_ubuntu_5k.jsonl" ]; then
    RAW_FILES+=("${RAW_ROOT}/agentnet_ubuntu_5k.jsonl")
fi

if [ ${#RAW_FILES[@]} -eq 0 ]; then
    echo "✗ Error: No raw AgentNet JSONL files found in ${RAW_ROOT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Merge raw trajectories (idempotent)
# ---------------------------------------------------------------------------

if [ ! -f "${MERGED_JSONL}" ] || [ "${FORCE_AGENTNET_MERGE:-0}" = "1" ]; then
    echo "Merging raw trajectories into ${MERGED_JSONL} ..."
    TMP_FILE="${MERGED_JSONL}.tmp"
    cat "${RAW_FILES[@]}" > "${TMP_FILE}"
    mv "${TMP_FILE}" "${MERGED_JSONL}"
else
    echo "Using existing ${MERGED_JSONL} (set FORCE_AGENTNET_MERGE=1 to rebuild)"
fi

MERGED_COUNT=$(wc -l < "${MERGED_JSONL}")
echo "✓ Combined trajectories: ${MERGED_COUNT}"
echo ""

# ---------------------------------------------------------------------------
# 3. Run transformation
# ---------------------------------------------------------------------------

TRANSFORM_CMD=(python)
if command -v uv >/dev/null 2>&1; then
    TRANSFORM_CMD=(uv run python)
fi

ARGS=(
    scripts/agentnet_scripts/transform_agentnet_for_training.py
    --input "${MERGED_JSONL}"
    --output-dir "${OUTPUT_DIR}"
    --base-image-dir "${IMAGES_DIR}"
    --val-ratio "${VAL_RATIO}"
    --test-ratio "${TEST_RATIO}"
    --min-traj-length "${MIN_TRAJ}"
    --seed "${SEED}"
)

if [ -n "${MAX_EXAMPLES}" ]; then
    ARGS+=(--max-examples-per-trajectory "${MAX_EXAMPLES}")
fi

if [ "${AGENTNET_INCLUDE_REFLECTION:-0}" = "1" ]; then
    ARGS+=(--include-reflection)
fi

echo "Running transformer:"
printf '  %q' "${TRANSFORM_CMD[@]}" "${ARGS[@]}"
echo ""

"${TRANSFORM_CMD[@]}" "${ARGS[@]}"

echo ""
echo "Transformation finished. Verifying splits..."

for SPLIT in train val test; do
    FILE="${OUTPUT_DIR}/agentnet_${SPLIT}.jsonl"
    if [ ! -f "${FILE}" ]; then
        echo "✗ Error: ${FILE} not found."
        exit 1
    fi
    COUNT=$(wc -l < "${FILE}")
    printf "✓ %-5s -> %s samples (%s)\n" "${SPLIT}" "${COUNT}" "${FILE}"
done

echo ""
echo "Sample record:"
SAMPLE_CMD=(python)
if command -v uv >/dev/null 2>&1; then
    SAMPLE_CMD=(uv run python)
fi

"${SAMPLE_CMD[@]}" - <<'PY'
import json
from pathlib import Path

sample_path = Path("data/agentnet_train.jsonl")
with sample_path.open("r", encoding="utf-8") as fh:
    line = fh.readline().strip()
    if line:
        obj = json.loads(line)
        preview = {
            "messages": obj.get("messages", [])[:2],
            "has_images": bool(obj.get("images")),
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
    else:
        print("{}")
PY

echo ""
echo "========================================"
echo "AgentNet data prep complete!"
echo "End time: $(date)"
echo "========================================"

