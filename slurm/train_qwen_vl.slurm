#!/bin/bash
#SBATCH --job-name=qwen3vl_finetune
#SBATCH --output=logs/train_qwen3vl_%j.out
#SBATCH --error=logs/train_qwen3vl_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G
#SBATCH --partition=gpu

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
WORKSPACE_DIR="/Users/asanshaygupta/Documents/Codes/websight-v2"
cd "${WORKSPACE_DIR}"

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

# Run training with DeepSpeed
# Using deepspeed launcher for multi-GPU training
deepspeed --num_gpus=8 --master_port=29500 \
    LLaMA-Factory/src/train.py \
    configs/qwen_vl_lora.yaml \
    --dataset_dir "${WORKSPACE_DIR}/data" \
    --dataset_info "${WORKSPACE_DIR}/configs/dataset_info.json"

# Alternative: Use accelerate launch if deepspeed is not available
# accelerate launch --config_file configs/accelerate_config.yaml \
#     LLaMA-Factory/src/train.py \
#     configs/qwen_vl_lora.yaml \
#     --dataset_dir "${WORKSPACE_DIR}/data" \
#     --dataset_info "${WORKSPACE_DIR}/configs/dataset_info.json"

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

