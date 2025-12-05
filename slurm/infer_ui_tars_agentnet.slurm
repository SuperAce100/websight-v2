#!/bin/bash
#SBATCH --job-name=infer_ui_tars
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/infer_ui_tars_%j.out
#SBATCH --error=logs/infer_ui_tars_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example inference on the merged UI-TARS AgentNet model.
# Expects a local image and runs a single prompt using transformers.

set -euo pipefail

WORKSPACE_DIR="/hai/users/a/s/asanshay/websight-v2"
cd "${WORKSPACE_DIR}"



echo "Workspace : ${WORKSPACE_DIR}"
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
if [ ! -d "venv" ]; then
    echo "✗ Error: venv not found in ${WORKSPACE_DIR}"
    exit 1
fi
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python scripts/infer_ui_tars_agentnet.py

echo ""
echo "End : $(date)"

