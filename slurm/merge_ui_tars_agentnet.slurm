#!/bin/bash
#SBATCH --job-name=merge_ui_tars_agentnet
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/merge_ui_tars_agentnet_%j.out
#SBATCH --error=logs/merge_ui_tars_agentnet_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Merge the latest UI-TARS AgentNet LoRA checkpoint into a full model.
# This uses llamafactory-cli export and writes results to scratch to avoid quota issues.
# Then it pushes the merged model to Hugging Face. HF_TOKEN must be set in the environment.

set -euo pipefail

WORKSPACE_DIR="/hai/users/a/s/asanshay/websight-v2"
cd "${WORKSPACE_DIR}"

echo "Workspace : ${WORKSPACE_DIR}"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : ${SLURM_NODELIST}"
echo "Start     : $(date)"
echo ""

# Paths
BASE_MODEL="ByteDance-Seed/UI-TARS-1.5-7B"
CKPT_ROOT="/hai/users/a/s/asanshay/websight-v2/saves/ui-tars-1.5-7b/lora/agentnet"
# Use the full saved adapter directory (contains adapter_model.safetensors), not a sub-checkpoint.
CKPT_PATH="${CKPT_ROOT}"
SCRATCH_ROOT="/hai/scratch/asanshay/websight-v2"
MERGE_OUT="${SCRATCH_ROOT}/merged/ui-tars-1.5-7b-agentnet-full"
HUB_MODEL_ID="Asanshay/websight-2-7B-kto"

if [ -z "${CKPT_PATH}" ]; then
    echo "✗ Error: no checkpoints found under ${CKPT_ROOT}"
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "✗ Error: HF_TOKEN must be set in the environment to push to Hugging Face."
    exit 1
fi

echo "Base model : ${BASE_MODEL}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Output dir : ${MERGE_OUT}"
echo ""

# Caches to scratch
CACHE_ROOT="${SCRATCH_ROOT}/.cache"
HF_HOME="${CACHE_ROOT}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE TRITON_CACHE_DIR
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRITON_CACHE_DIR}"

# Env / deps
if [ ! -d ".venv" ]; then
    echo "✗ Error: .venv not found in ${WORKSPACE_DIR}"
    exit 1
fi
source .venv/bin/activate

if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "✗ Error: llamafactory-cli not found (install via uv pip -e LLaMA-Factory)"
    exit 1
fi
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "✗ Error: huggingface-cli not found. Install with: uv pip install huggingface_hub"
    exit 1
fi

# Merge
mkdir -p "${MERGE_OUT}"
echo "Running merge with llamafactory-cli export..."
llamafactory-cli export \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_name_or_path "${CKPT_PATH}" \
    --export_dir "${MERGE_OUT}" \
    --template qwen2_vl \
    --finetuning_type lora \
    --export_size 2 \
    --export_legacy_format False

echo ""
echo "✓ Merge complete. Artifacts:"
ls -lh "${MERGE_OUT}"

# Optional: push to Hugging Face if HUB_MODEL_ID is provided
echo ""
echo "Pushing to Hugging Face: ${HUB_MODEL_ID}"
huggingface-cli upload "${MERGE_OUT}" "${HUB_MODEL_ID}" \
    --repo-type model \
    --token "${HF_TOKEN}" \
    --exclude ".git/*"
echo "✓ Push complete: https://huggingface.co/${HUB_MODEL_ID}"

echo "End   : $(date)"

