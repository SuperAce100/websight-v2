#!/bin/bash
#SBATCH --job-name=qwen3vl_noeval
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/train_noeval_%j.out
#SBATCH --error=logs/train_noeval_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G

# Qwen3-VL-8B Fine-tuning - NO EVALUATION
# Evaluation was hanging and taking 2+ hours
# This version: TRAIN ONLY, save checkpoints, skip eval entirely

set -e
set -u

echo "========================================"
echo "Qwen3-VL Training (No Evaluation)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Start time: $(date)"
echo ""

WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

DATA_DIR="/hai/scratch/asanshay/websight-v2/data"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORKSPACE_DIR}/.cache/huggingface/datasets"
export CUDA_TF32_ENABLED=1
export TRITON_CACHE_DIR="${WORKSPACE_DIR}/.cache/triton"
mkdir -p "${TRITON_CACHE_DIR}"

# Create directories
mkdir -p logs
mkdir -p saves/qwen3-vl-8b/lora/sft-noeval
mkdir -p .cache/huggingface

# Print system info
echo ""
echo "========================================"
echo "System Information"
echo "========================================"
nvidia-smi
echo ""
python --version
echo ""

# Verify data
echo "========================================"
echo "Checking data files..."
echo "========================================"

if [ ! -f "data/wave_ui_train.jsonl" ]; then
    echo "✗ Error: Training data not found"
    exit 1
fi

TRAIN_COUNT=$(wc -l < data/wave_ui_train.jsonl)
echo "✓ Training samples: ${TRAIN_COUNT}"
echo ""

# Check LLaMA-Factory
if [ ! -d "LLaMA-Factory" ]; then
    echo "✗ Error: LLaMA-Factory not found"
    exit 1
fi
echo "✓ LLaMA-Factory found"
echo ""

# Verify dependencies
echo "========================================"
echo "Verifying dependencies..."
echo "========================================"
pip install --upgrade pip
pip install -q transformers>=4.57.0 || pip install -q git+https://github.com/huggingface/transformers
pip install -q accelerate 'deepspeed>=0.10.0,<=0.16.9'
pip install -q flash-attn --no-build-isolation
echo "✓ Dependencies verified"
echo ""

# Link dataset_info.json
mkdir -p LLaMA-Factory/data
mkdir -p data
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" LLaMA-Factory/data/dataset_info.json
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" data/dataset_info.json
echo "✓ Linked dataset_info.json"

# Start training
echo ""
echo "========================================"
echo "Starting training (NO EVALUATION)..."
echo "========================================"
echo "Configuration:"
echo "  Model: Qwen/Qwen3-VL-8B-Instruct"
echo "  LoRA: rank=32, alpha=64"
echo "  GPUs: 8xH100"
echo "  Batch: 4 per device × 2 accum × 8 GPUs = 64 effective"
echo "  Epochs: 1.0"
echo "  Save: Every 200 steps"
echo "  Eval: DISABLED (was causing hangs)"
echo ""
echo "Expected time: ~6-7 hours (no eval overhead)"
echo ""

# Train with NO evaluation
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --use_rslora \
    --dataset wave_ui_dataset \
    --dataset_dir "${WORKSPACE_DIR}/data" \
    --template qwen2_vl \
    --media_dir /hai/scratch/asanshay/websight-v2/data \
    --cutoff_len 8192 \
    --preprocessing_num_workers 16 \
    --output_dir saves/qwen3-vl-8b/lora/sft-noeval \
    --logging_steps 10 \
    --save_steps 200 \
    --plot_loss \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 1.0 \
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
    --flash_attn fa2

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Model saved to: saves/qwen3-vl-8b/lora/sft-noeval"
    echo ""
    echo "Checkpoints:"
    ls -lh saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-* 2>/dev/null || echo "  (will appear during training)"
    echo ""
    echo "To merge and push:"
    echo "  # Edit slurm/merge_and_push.slurm:"
    echo "  ADAPTER_PATH=\"saves/qwen3-vl-8b/lora/sft-noeval\""
    echo "  # Then:"
    echo "  sbatch --account=ingrai slurm/merge_and_push.slurm"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

