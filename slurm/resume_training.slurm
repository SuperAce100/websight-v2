#!/bin/bash
#SBATCH --job-name=qwen3vl_resume
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/resume_training_%j.out
#SBATCH --error=logs/resume_training_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G

# Resume training from checkpoint-200
# This will continue from where we left off (20% complete)

set -e
set -u

echo "========================================"
echo "Resuming Qwen3-VL Training from Checkpoint"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

# Find the latest checkpoint
CHECKPOINT_DIR=$(ls -td saves/qwen3-vl-8b/lora/sft-rank32/checkpoint-* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "❌ Error: No checkpoint found!"
    echo "Looking in: saves/qwen3-vl-8b/lora/sft-rank32/"
    ls -lh saves/qwen3-vl-8b/lora/sft-rank32/ || true
    exit 1
fi

echo "✓ Found checkpoint: ${CHECKPOINT_DIR}"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Environment setup
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
mkdir -p LLaMA-Factory/data
mkdir -p data

ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" LLaMA-Factory/data/dataset_info.json
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" data/dataset_info.json

echo "========================================"
echo "Resuming training..."
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Remaining: ~80% of training"
echo "Expected time: ~5-6 hours"
echo ""

# Resume training with optimized eval settings
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --resume_from_checkpoint "${CHECKPOINT_DIR}" \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --use_rslora \
    --dataset wave_ui_dataset \
    --eval_dataset wave_ui_val \
    --dataset_dir "${WORKSPACE_DIR}/data" \
    --template qwen2_vl \
    --media_dir /hai/scratch/asanshay/websight-v2/data \
    --cutoff_len 8192 \
    --preprocessing_num_workers 16 \
    --output_dir saves/qwen3-vl-8b/lora/sft-rank32 \
    --logging_steps 10 \
    --save_steps 200 \
    --plot_loss \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 \
    --ddp_timeout 180000000 \
    --eval_strategy steps \
    --eval_steps 800 \
    --max_eval_samples 1000 \
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
    echo "Model saved to: saves/qwen3-vl-8b/lora/sft-rank32"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

