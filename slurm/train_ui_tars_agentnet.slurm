#!/bin/bash
#SBATCH --job-name=ui_tars_agentnet
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/train_ui_tars_agentnet_%j.out
#SBATCH --error=logs/train_ui_tars_agentnet_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G

# LoRA fine-tuning of UI-TARS-1.5-7B on the AgentNet dataset with LLaMA-Factory.
# - Expects data generated via slurm/prepare_agentnet.slurm (ShareGPT format)
# - Uses uv for environment management per repo conventions
# - Runs on 8×H100 (adjust SBATCH directives for smaller jobs if needed)

set -euo pipefail

echo "========================================"
echo "UI-TARS AgentNet Fine-tune"
echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURM_NODELIST}"
echo "GPUs   : ${SLURM_GPUS_ON_NODE}"
echo "Start  : $(date)"
echo ""

WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

mkdir -p logs saves

DATA_DIR="${WORKSPACE_DIR}/data"
MEDIA_DIR="/hai/scratch/asanshay/websight-v2/agentnet/images"
TRAIN_JSON="${DATA_DIR}/agentnet_train.jsonl"
VAL_JSON="${DATA_DIR}/agentnet_val.jsonl"
TOKENIZED_PATH="/hai/scratch/asanshay/websight-v2/agentnet/tokenized/ui_tars_agentnet"
mkdir -p "${DATA_DIR}" "$(dirname "${TOKENIZED_PATH}")"

CONFIG_PATH="configs/train_agentnet_lora.yaml"
CACHE_ROOT="/hai/scratch/asanshay/websight-v2/.cache"
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
echo "Tokenized: ${TOKENIZED_PATH}"
echo ""

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------

if [ ! -f "${TRAIN_JSON}" ]; then
    echo "✗ Error: ${TRAIN_JSON} not found."
    echo "  Run: sbatch slurm/prepare_agentnet.slurm"
    exit 1
fi

if [ ! -d "${MEDIA_DIR}" ]; then
    echo "✗ Error: Screenshot directory ${MEDIA_DIR} missing."
    echo "  Ensure AgentNet images are extracted there."
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
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
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

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "✗ Error: Config file ${CONFIG_PATH} not found."
    exit 1
fi

if [ ! -f "${WORKSPACE_DIR}/configs/deepspeed_zero2.json" ]; then
    echo "✗ Error: configs/deepspeed_zero2.json missing."
    exit 1
fi

echo "========================================"
echo "Starting llamafactory-cli train via ${CONFIG_PATH}"
echo "========================================"

llamafactory-cli train "${CONFIG_PATH}" --tokenized_path "${TOKENIZED_PATH}"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully."
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

