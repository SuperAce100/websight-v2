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
#SBATCH --cpus-per-task=12
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

DEFAULT_PROCESSED_DIR="${AGENTNET_PROCESSED_DIR:-${WORKSPACE_DIR}/data}"
DATA_DIR="${DEFAULT_PROCESSED_DIR}"
mkdir -p "${DATA_DIR}"

DEFAULT_MEDIA_DIR="/hai/scratch/asanshay/websight-v2/agentnet/images"
MEDIA_DIR="${AGENTNET_IMAGE_DIR:-${DEFAULT_MEDIA_DIR}}"
TRAIN_JSON="${DATA_DIR}/agentnet_train.jsonl"
VAL_JSON="${DATA_DIR}/agentnet_val.jsonl"
TEST_JSON="${DATA_DIR}/agentnet_test.jsonl"

BASE_MODEL="${BASE_MODEL_PATH:-ByteDance-Seed/UI-TARS-1.5-7B}"
MODEL_REVISION="${BASE_MODEL_REVISION:-main}"
TEMPLATE="${UI_TARS_TEMPLATE:-ui_tars}"  # Set to sharegpt if template is unavailable
OUTPUT_DIR="${WORKSPACE_DIR}/saves/ui-tars-1.5-7b/lora/agentnet"
mkdir -p "${OUTPUT_DIR}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM_STEPS:-4}"
EPOCHS="${AGENTNET_EPOCHS:-1.0}"
LR="${AGENTNET_LR:-5e-5}"
LORA_RANK="${AGENTNET_LORA_RANK:-64}"
LORA_ALPHA="${AGENTNET_LORA_ALPHA:-128}"
LORA_DROPOUT="${AGENTNET_LORA_DROPOUT:-0.05}"
SAVE_STEPS="${AGENTNET_SAVE_STEPS:-100}"
EVAL_STEPS="${AGENTNET_EVAL_STEPS:-400}"
WORKERS="${AGENTNET_DATA_WORKERS:-16}"
CUTOFF_LEN="${AGENTNET_CUTOFF_LEN:-8192}"
MAX_SAMPLES="${AGENTNET_MAX_SAMPLES:-450000}"
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
echo "Model    : ${BASE_MODEL} (rev: ${MODEL_REVISION})"
echo "Template : ${TEMPLATE}"
echo "Output   : ${OUTPUT_DIR}"
echo "Media dir: ${MEDIA_DIR}"
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
    echo "  Ensure AgentNet images are extracted there or override AGENTNET_IMAGE_DIR."
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
export TOKENIZERS_PARALLELISM=false
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

TRAIN_COUNT=$(wc -l < "${TRAIN_JSON}")
echo "✓ Training samples: ${TRAIN_COUNT}"
if [ -f "${VAL_JSON}" ]; then
    VAL_COUNT=$(wc -l < "${VAL_JSON}")
    echo "✓ Validation samples: ${VAL_COUNT}"
else
    echo "⚠ No validation split detected (agentnet_val.jsonl missing). Eval will be skipped."
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Build CLI arguments
# ---------------------------------------------------------------------------

EVAL_ARGS=()
if [ -f "${VAL_JSON}" ]; then
    EVAL_ARGS+=(
        --do_eval
        --eval_dataset agentnet_val
        --per_device_eval_batch_size "${PER_DEVICE_BATCH}"
        --eval_strategy steps
        --eval_steps "${EVAL_STEPS}"
    )
else
    EVAL_ARGS+=(--eval_strategy "no")
fi

MEDIA_ARGS=()
if [ -d "${MEDIA_DIR}" ]; then
    MEDIA_ARGS+=(--media_dir "${MEDIA_DIR}")
fi

if [ ! -f "${WORKSPACE_DIR}/configs/deepspeed_zero2.json" ]; then
    echo "✗ Error: configs/deepspeed_zero2.json missing."
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Launch training
# ---------------------------------------------------------------------------

echo "========================================"
echo "Starting llamafactory-cli train"
echo "========================================"
echo "Per-device batch : ${PER_DEVICE_BATCH}"
echo "Grad accumulation: ${GRAD_ACCUM}"
echo "Effective batch  : $((PER_DEVICE_BATCH * GRAD_ACCUM * 8))"
echo "Epochs           : ${EPOCHS}"
echo "Learning rate    : ${LR}"
echo "LoRA rank/alpha  : ${LORA_RANK}/${LORA_ALPHA}"
echo "Max samples      : ${MAX_SAMPLES}"
echo ""

MAX_SAMPLE_ARGS=()
if [ "${MAX_SAMPLES}" != "-1" ]; then
    MAX_SAMPLE_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

llamafactory-cli train \
    --model_name_or_path "${BASE_MODEL}" \
    --model_revision "${MODEL_REVISION}" \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --use_rslora \
    --dataset agentnet_train \
    --dataset_dir "${DATA_DIR}" \
    --template "${TEMPLATE}" \
    "${MEDIA_ARGS[@]}" \
    --cutoff_len "${CUTOFF_LEN}" \
    "${MAX_SAMPLE_ARGS[@]}" \
    --preprocessing_num_workers "${WORKERS}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 10 \
    --save_steps "${SAVE_STEPS}" \
    --plot_loss \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --num_train_epochs "${EPOCHS}" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --save_strategy steps \
    --save_total_limit 3 \
    --deepspeed configs/deepspeed_zero2.json \
    --report_to tensorboard \
    --gradient_checkpointing \
    --flash_attn fa2 \
    "${EVAL_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully."
    echo "Adapter path: ${OUTPUT_DIR}"
    echo "Next: edit slurm/merge_and_push.slurm to merge & upload weights."
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

