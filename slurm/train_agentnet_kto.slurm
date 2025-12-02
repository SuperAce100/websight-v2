#!/bin/bash
#SBATCH --job-name=agentnet_kto
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/train_agentnet_kto_%j.out
#SBATCH --error=logs/train_agentnet_kto_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G

# KTO (Kahneman-Tversky Optimization) fine-tuning of UI-TARS-1.5-7B on AgentNet dataset
# - Starts from SFT checkpoint: saves/ui-tars-1.5-7b/lora/agentnet
# - Expects KTO data generated via scripts/data_prep/prepare_kto_data.sh
# - Uses uv for environment management per repo conventions
# - Runs on 8×H100 GPUs

set -euo pipefail

echo "========================================"
echo "UI-TARS AgentNet KTO Fine-tune"
echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURM_NODELIST}"
echo "GPUs   : ${SLURM_GPUS_ON_NODE}"
echo "Start  : $(date)"
echo ""

WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

mkdir -p logs saves

DEFAULT_PROCESSED_DIR="${AGENTNET_PROCESSED_DIR:-${WORKSPACE_DIR}/data}"
DATA_DIR="${DEFAULT_PROCESSED_DIR}"
mkdir -p "${DATA_DIR}"

DEFAULT_MEDIA_DIR="/hai/scratch/asanshay/websight-v2/agentnet/images"
MEDIA_DIR="${AGENTNET_IMAGE_DIR:-${DEFAULT_MEDIA_DIR}}"
KTO_JSON="${DATA_DIR}/agentnet_kto.jsonl"

CONFIG_PATH="${LLF_CONFIG_PATH:-configs/train_agentnet_kto.yaml}"
DEFAULT_CACHE_ROOT="/hai/scratch/asanshay/websight-v2/.cache"
CACHE_ROOT="${AGENTNET_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
UV_CACHE_DIR="${CACHE_ROOT}/uv"
PIP_CACHE_DIR="${CACHE_ROOT}/pip"
XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
TMPDIR="${CACHE_ROOT}/tmp"
export UV_CACHE_DIR PIP_CACHE_DIR XDG_CACHE_HOME TMPDIR
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

echo "Workspace: ${WORKSPACE_DIR}"
echo "Data dir : ${DATA_DIR}"
echo "Config   : ${CONFIG_PATH}"
echo "Media dir: ${MEDIA_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------

if [ ! -f "${KTO_JSON}" ]; then
    echo "✗ Error: ${KTO_JSON} not found."
    echo "  Run: bash scripts/data_prep/prepare_kto_data.sh"
    echo "  Or: sbatch slurm/prepare_agentnet.slurm (if you need to create merged file first)"
    exit 1
fi

if [ ! -d "${MEDIA_DIR}" ]; then
    echo "✗ Error: Screenshot directory ${MEDIA_DIR} missing."
    echo "  Ensure AgentNet images are extracted there or override AGENTNET_IMAGE_DIR."
    exit 1
fi

# Check SFT checkpoint exists
SFT_CHECKPOINT="${WORKSPACE_DIR}/saves/ui-tars-1.5-7b/lora/agentnet"
if [ ! -d "${SFT_CHECKPOINT}" ]; then
    echo "✗ Error: SFT checkpoint not found at ${SFT_CHECKPOINT}"
    echo "  KTO training requires a pre-trained SFT checkpoint."
    echo "  Run SFT training first: sbatch slurm/train_ui_tars_agentnet.slurm"
    exit 1
fi

if [ ! -d "LLaMA-Factory" ]; then
    cat <<EOF
✗ Error: LLaMA-Factory directory not found.
Clone it in the repo root before running:
  git clone https://github.com/hiyouga/LLaMA-Factory.git
  uv pip install -e LLaMA-Factory
EOF
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "✗ Error: uv is required but not found in PATH."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Environment setup (uv + .venv)
# ---------------------------------------------------------------------------

PYTHON_VERSION="${UI_TARS_PYTHON:-3.12}"
if [ ! -d ".venv" ]; then
    echo "Creating uv virtual environment (.venv)..."
    uv venv --python "${PYTHON_VERSION}" .venv
fi

source .venv/bin/activate

echo "Installing training dependencies with uv pip..."
uv pip install --upgrade pip
uv pip install -r requirements-training.txt
uv pip install flash-attn --no-build-isolation
uv pip install -e LLaMA-Factory

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORKSPACE_DIR}/.cache/huggingface/datasets"
export TRITON_CACHE_DIR="${WORKSPACE_DIR}/.cache/triton"
export CUDA_TF32_ENABLED=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"

echo ""
echo "========================================"
echo "System Information"
echo "========================================"
nvidia-smi
python --version
echo ""

# ---------------------------------------------------------------------------
# 3. Link dataset metadata
# ---------------------------------------------------------------------------

mkdir -p LLaMA-Factory/data
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" LLaMA-Factory/data/dataset_info.json
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" "${DATA_DIR}/dataset_info.json"
echo "✓ Linked dataset_info.json"

KTO_COUNT=$(wc -l < "${KTO_JSON}")
echo "✓ KTO training samples: ${KTO_COUNT}"
echo "✓ SFT checkpoint: ${SFT_CHECKPOINT}"
echo ""

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "✗ Error: Config file ${CONFIG_PATH} not found."
    exit 1
fi

if [ ! -f "${WORKSPACE_DIR}/configs/deepspeed_zero2.json" ]; then
    echo "✗ Error: configs/deepspeed_zero2.json missing."
    exit 1
fi

echo "========================================"
echo "Starting KTO training via ${CONFIG_PATH}"
echo "========================================"
echo "Method: KTO (Kahneman-Tversky Optimization)"
echo "Base model: UI-TARS-1.5-7B"
echo "Starting from: SFT checkpoint"
echo "Dataset: agentnet_kto_train"
echo ""

llamafactory-cli train "${CONFIG_PATH}"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ KTO training completed successfully."
    echo "Adapter path: saves/ui-tars-1.5-7b/lora/agentnet-kto"
    echo "Next: edit slurm/merge_and_push.slurm to merge & upload weights."
else
    echo "✗ KTO training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

