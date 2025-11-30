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
# - Downloads (optional) and merges Windows/macOS + Ubuntu trajectories
# - Automatically unpacks split screenshot archives
# - Runs scripts/agentnet_scripts/transform_agentnet_for_training.py to emit ShareGPT jsonl

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
AUTO_UNZIP="${AGENTNET_AUTO_UNZIP:-1}"
FORCE_UNZIP="${FORCE_AGENTNET_UNZIP:-0}"

# Use scratch-backed caches/temp to avoid AFS quota issues
DEFAULT_CACHE_ROOT="/hai/scratch/asanshay/websight-v2/.cache"
CACHE_ROOT="${AGENTNET_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
UV_CACHE_DIR="${CACHE_ROOT}/uv"
TMPDIR="${CACHE_ROOT}/tmp"
export UV_CACHE_DIR TMPDIR
mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}"

echo "Workspace: ${WORKSPACE_DIR}"
echo "Raw dir : ${RAW_ROOT}"
echo "Images  : ${IMAGES_DIR}"
echo "Output  : ${OUTPUT_DIR}"
echo "Cache   : ${CACHE_ROOT}"
echo ""

ensure_hf_tooling() {
    if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
        return 0
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "✗ Error: Hugging Face CLI not found and uv is unavailable to install it."
        return 1
    fi

    echo "Hugging Face CLI tools not found. Installing huggingface_hub via uv..."
    uv tool install huggingface_hub

    if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
        return 0
    fi

    echo "✗ Error: Failed to install Hugging Face CLI tools via uv."
    return 1
}

run_hf_download() {
    ensure_hf_tooling || return 1

    HF_ARGS=(
        --repo-type dataset
        --local-dir "${RAW_ROOT}"
        --local-dir-use-symlinks False
    )
    if [ -n "${HF_TOKEN:-}" ]; then
        HF_ARGS+=(--token "${HF_TOKEN}")
    fi

    if command -v hf >/dev/null 2>&1; then
        echo "Using 'hf download' to fetch xlangai/AgentNet -> ${RAW_ROOT}"
        hf download "${HF_ARGS[@]}" xlangai/AgentNet
    else
        echo "⚠ Using legacy 'huggingface-cli download' (deprecated upstream)"
        huggingface-cli download "${HF_ARGS[@]}" xlangai/AgentNet
    fi
}

has_pngs() {
    local dir="$1"
    if [ ! -d "${dir}" ]; then
        return 1
    fi
    if find "${dir}" -type f -name '*.png' -print -quit 2>/dev/null | grep -q .; then
        return 0
    fi
    return 1
}

merge_split_archive() {
    local split_dir="$1"
    local label="$2"
    local base_zip="${split_dir}/images.zip"

    if [ ! -f "${base_zip}" ]; then
        return 1
    fi

    mkdir -p "${IMAGES_DIR}"
    local merged_zip="${split_dir}/${label}_images_full.zip"
    if [ "${FORCE_UNZIP}" = "1" ] || [ ! -f "${merged_zip}" ]; then
        echo "→ Merging split archive for ${label} (this may take a while)..."
        (cd "${split_dir}" && zip -s 0 images.zip --out "$(basename "${merged_zip}")")
    else
        echo "→ Reusing existing merged archive for ${label}"
    fi

    echo "→ Extracting ${label} screenshots into ${IMAGES_DIR}"
    unzip -oq "${merged_zip}" -d "${IMAGES_DIR}"
}

ensure_images_available() {
    if has_pngs "${IMAGES_DIR}"; then
        return 0
    fi

    if [ "${AUTO_UNZIP}" = "1" ]; then
        echo "Screenshots not found at ${IMAGES_DIR}. Attempting to unpack split archives..."
        merge_split_archive "${RAW_ROOT}/win_mac_images" "win_mac" || true
        merge_split_archive "${RAW_ROOT}/ubuntu_images" "ubuntu" || true
    fi

    if has_pngs "${IMAGES_DIR}"; then
        echo "✓ Screenshots extracted to ${IMAGES_DIR}"
        return 0
    fi

    cat <<EOF
✗ Error: Screenshot directory not found at ${IMAGES_DIR}

Automatic extraction failed. If the dataset contains split archives, you can run:
  cd ${RAW_ROOT}/win_mac_images
  zip -s 0 images.zip --out images_win_mac_full.zip
  unzip -n images_win_mac_full.zip -d ${IMAGES_DIR}

  cd ${RAW_ROOT}/ubuntu_images
  zip -s 0 images.zip --out images_ubuntu_full.zip
  unzip -n images_ubuntu_full.zip -d ${IMAGES_DIR}

Ensure all extracted *.png files reside under ${IMAGES_DIR}, then re-run this job.
EOF
    return 1
}

if [ ! -d "${RAW_ROOT}" ]; then
    if [ "${AUTO_DOWNLOAD}" != "1" ]; then
        cat <<EOF
✗ Error: ${RAW_ROOT} not found.
Either download xlangai/AgentNet manually or set AGENTNET_AUTO_DOWNLOAD=1 before submitting.
EOF
        exit 1
    fi

    echo "Downloading AgentNet dataset to ${RAW_ROOT} ..."
    mkdir -p "${RAW_ROOT}"
    run_hf_download || exit 1
fi

mkdir -p "${OUTPUT_DIR}"

ensure_images_available || exit 1

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