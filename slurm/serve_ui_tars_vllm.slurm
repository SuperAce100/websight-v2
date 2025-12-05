#!/bin/bash
#SBATCH --job-name=serve_ui_tars_vllm
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/serve_ui_tars_vllm_%j.out
#SBATCH --error=logs/serve_ui_tars_vllm_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# Serve the merged UI-TARS AgentNet model via vLLM OpenAI-compatible API.
# Model path and caches are fixed to scratch to avoid quota issues.
# Requires vLLM installed in the Python environment.

set -euo pipefail

WORKSPACE_DIR="/hai/users/a/s/asanshay/websight-v2"
cd "${WORKSPACE_DIR}"

MODEL_PATH="/hai/scratch/asanshay/websight-v2/merged/ui-tars-1.5-7b-agentnet-full"
HOST="0.0.0.0"
PORT="8000"
TP="8"
MAX_MODEL_LEN="4096"

echo "Workspace : ${WORKSPACE_DIR}"
echo "Model     : ${MODEL_PATH}"
echo "Host/Port : ${HOST}:${PORT}"
echo "TensorPar : ${TP}"
echo "Start     : $(date)"
echo ""

# Caches to scratch
SCRATCH_ROOT="/hai/scratch/asanshay/websight-v2"
CACHE_ROOT="${SCRATCH_ROOT}/.cache"
HF_HOME="${CACHE_ROOT}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE TRITON_CACHE_DIR
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRITON_CACHE_DIR}"

# Env
if [ ! -d ".venv" ]; then
    echo "✗ Error: .venv not found in ${WORKSPACE_DIR}"
    exit 1
fi
source .venv/bin/activate

if ! command -v python >/dev/null 2>&1; then
    echo "✗ Error: python not found"
    exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1; then
    import importlib, sys
    sys.exit(1 if importlib.util.find_spec("vllm") is None else 0)
PY
then
    echo "✗ Error: vllm not installed in this environment."
    echo "Install with: uv pip install 'vllm>=0.4.2'"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Launching vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${TP}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype float16 \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.9 \
    --swap-space 8 \
    --disable-log-requests

echo ""
echo "Server stopped. End: $(date)"

