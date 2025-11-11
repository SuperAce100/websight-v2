#!/bin/bash
#SBATCH --job-name=qwen3vl_rank32
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/train_qwen3vl_rank32_%j.out
#SBATCH --error=logs/train_qwen3vl_rank32_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G

# Qwen3-VL-8B Fine-tuning on 8xH100 GPUs - OPTIMIZED FOR SPEED
# Uses lower LoRA rank (32 vs 64) for faster training while using ALL samples
# Training time: ~6-8 hours
# Expected to complete within 8-hour limit

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Qwen3-VL Optimized Training (Rank 32)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Start time: $(date)"
echo ""

# Set workspace directory
WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

# Dataset location
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

# Set TRITON_CACHE_DIR to avoid NFS warning
export TRITON_CACHE_DIR="${WORKSPACE_DIR}/.cache/triton"
mkdir -p "${TRITON_CACHE_DIR}"

# Create necessary directories
mkdir -p logs
mkdir -p saves/qwen3-vl-8b/lora/sft-rank32
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
    echo "✗ Error: LLaMA-Factory directory not found!"
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
pip install -q accelerate 'deepspeed>=0.10.0,<=0.16.9'
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

# Link dataset_info.json to both LLaMA-Factory and data directories
mkdir -p LLaMA-Factory/data
mkdir -p data
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" LLaMA-Factory/data/dataset_info.json
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" data/dataset_info.json
echo "✓ Linked dataset_info.json to LLaMA-Factory/data/ and data/"

# Start training
echo ""
echo "========================================"
echo "Starting OPTIMIZED training..."
echo "========================================"
echo "Configuration:"
echo "  Model: Qwen/Qwen3-VL-8B-Instruct"
echo "  Method: LoRA (rank=32, alpha=64) - REDUCED FOR SPEED"
echo "  GPUs: 8xH100"
echo "  Batch size: 4 per device × 2 grad accum × 8 GPUs = effective 64"
echo "  Epochs: 1.0"
echo "  Learning rate: 5e-5"
echo "  Save steps: 200"
echo "  Max samples: ALL (no limit)"
echo ""
echo "Speed optimizations:"
echo "  ✓ LoRA rank: 32 (vs 64) - ~40% faster training"
echo "  ✓ LoRA alpha: 64 (vs 128) - matched to rank"
echo "  ✓ Larger batch: 4 (vs 2) - 2x throughput"
echo "  ✓ Fewer epochs: 1.0 (vs 2.5) - 2.5x faster"
echo "  ✓ Frequent saves: 200 steps - more checkpoints"
echo "  ✓ All samples: No artificial limit"
echo ""
echo "Expected completion: 6-8 hours"
echo ""

# Run training with optimized settings
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
    --eval_steps 200 \
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
    echo "Model saved to: saves/qwen3-vl-8b/lora/sft-rank32"
    echo ""
    echo "To merge LoRA weights with base model:"
    echo "  llamafactory-cli export \\"
    echo "    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \\"
    echo "    --adapter_name_or_path saves/qwen3-vl-8b/lora/sft-rank32 \\"
    echo "    --export_dir saves/qwen3-vl-8b-merged \\"
    echo "    --template qwen2_vl \\"
    echo "    --finetuning_type lora \\"
    echo "    --export_size 2 \\"
    echo "    --export_legacy_format False"
    echo ""
    echo "Or push to HuggingFace:"
    echo "  # Edit slurm/merge_and_push.slurm and change:"
    echo "  ADAPTER_PATH=\"saves/qwen3-vl-8b/lora/sft-rank32\""
    echo "  # Then run:"
    echo "  sbatch --account=ingrai slurm/merge_and_push.slurm"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

