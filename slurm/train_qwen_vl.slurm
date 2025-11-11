#!/bin/bash
#SBATCH --job-name=qwen3vl_finetune
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/train_qwen3vl_%j.out
#SBATCH --error=logs/train_qwen3vl_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G

# Qwen3-VL-8B Fine-tuning on 8xH100 GPUs
# Training time: up to 8 hours
# Expected throughput: ~10-15 samples/sec with batch size 2, grad accum 4

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Qwen3-VL Fine-tuning Training Job"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Start time: $(date)"
echo ""

# Load required modules (adjust based on your cluster)
# Uncomment and modify these lines based on your HPC environment
# module load cuda/12.1
# module load cudnn/8.9
# module load nccl/2.18
# module load python/3.10

# Set workspace directory
WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

# Dataset location
DATA_DIR="/hai/scratch/asanshay/websight-v2/data"

# Activate virtual environment
# Adjust this path to your actual virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Environment variables for optimal H100 performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORKSPACE_DIR}/.cache/huggingface/datasets"

# Enable TF32 for better performance on H100
export CUDA_TF32_ENABLED=1

# Create necessary directories
mkdir -p logs
mkdir -p saves/qwen3-vl-8b/lora/sft
mkdir -p .cache/huggingface

# Print system information
echo ""
echo "========================================"
echo "System Information"
echo "========================================"
nvidia-smi
echo ""
python --version
echo ""

# Verify data files exist
echo "========================================"
echo "Checking data files..."
echo "========================================"

if [ ! -f "data/wave_ui_train.jsonl" ]; then
    echo "✗ Error: Training data not found at data/wave_ui_train.jsonl"
    echo "Please run the data preparation job first:"
    echo "  sbatch slurm/prepare_data.slurm"
    exit 1
fi

if [ ! -f "data/wave_ui_val.jsonl" ]; then
    echo "✗ Error: Validation data not found at data/wave_ui_val.jsonl"
    echo "Please run the data preparation job first:"
    echo "  sbatch slurm/prepare_data.slurm"
    exit 1
fi

TRAIN_COUNT=$(wc -l < data/wave_ui_train.jsonl)
VAL_COUNT=$(wc -l < data/wave_ui_val.jsonl)
echo "✓ Training samples: ${TRAIN_COUNT}"
echo "✓ Validation samples: ${VAL_COUNT}"
echo ""

# Check if LLaMA-Factory is available
echo "========================================"
echo "Checking LLaMA-Factory..."
echo "========================================"

if [ ! -d "LLaMA-Factory" ]; then
    echo "✗ Error: LLaMA-Factory not found!"
    echo "Please clone LLaMA-Factory:"
    echo "  git clone https://github.com/hiyouga/LLaMA-Factory.git"
    echo "  cd LLaMA-Factory"
    echo "  pip install -e ."
    exit 1
fi

echo "✓ LLaMA-Factory found"
echo ""

# Install/verify dependencies
echo "========================================"
echo "Verifying dependencies..."
echo "========================================"

pip install --upgrade pip
pip install -q transformers>=4.57.0 || pip install -q git+https://github.com/huggingface/transformers
pip install -q accelerate deepspeed
pip install -q flash-attn --no-build-isolation

# Navigate to LLaMA-Factory if not already installed
if [ ! -f "LLaMA-Factory/src/train.py" ] && [ -d "LLaMA-Factory" ]; then
    echo "Installing LLaMA-Factory..."
    cd LLaMA-Factory
    pip install -e .
    cd ..
fi

echo "✓ Dependencies verified"
echo ""

# Start training
echo "========================================"
echo "Starting training..."
echo "========================================"
echo "Configuration: configs/qwen_vl_lora.yaml"
echo "Dataset: wave_ui_dataset"
echo "Model: Qwen/Qwen3-VL-8B-Instruct"
echo "Method: LoRA (rank=64, alpha=128)"
echo "GPUs: 8xH100"
echo "Batch size: 2 per device × 4 grad accum × 8 GPUs = effective 64"
echo ""

# Set TRITON_CACHE_DIR to avoid NFS warning
export TRITON_CACHE_DIR="${WORKSPACE_DIR}/.cache/triton"
mkdir -p "${TRITON_CACHE_DIR}"

# Link dataset_info.json to both LLaMA-Factory and data directories
mkdir -p LLaMA-Factory/data
mkdir -p data
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" LLaMA-Factory/data/dataset_info.json
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" data/dataset_info.json
echo "✓ Linked dataset_info.json to LLaMA-Factory/data/ and data/"

# Run training with LLaMA-Factory CLI
# Using llamafactory-cli with DeepSpeed for multi-GPU training
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --use_rslora \
    --dataset wave_ui_dataset \
    --eval_dataset wave_ui_val \
    --dataset_dir "${WORKSPACE_DIR}/data" \
    --template qwen2_vl \
    --media_dir /hai/scratch/asanshay/websight-v2/data \
    --cutoff_len 8192 \
    --max_samples 100000 \
    --preprocessing_num_workers 16 \
    --output_dir saves/qwen3-vl-8b/lora/sft \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 2.5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 \
    --ddp_timeout 180000000 \
    --eval_strategy steps \
    --eval_steps 500 \
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
    echo "Model saved to: saves/qwen3-vl-8b/lora/sft"
    echo ""
    echo "To merge LoRA weights with base model:"
    echo "  python LLaMA-Factory/src/export_model.py \\"
    echo "    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \\"
    echo "    --adapter_name_or_path saves/qwen3-vl-8b/lora/sft \\"
    echo "    --export_dir saves/qwen3-vl-8b-merged \\"
    echo "    --export_size 2 \\"
    echo "    --export_device cpu"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

