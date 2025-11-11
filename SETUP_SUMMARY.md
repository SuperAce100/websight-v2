# Qwen3-VL Fine-tuning Setup - Summary

## ✓ Setup Complete

All scripts and configurations for fine-tuning Qwen3-VL-8B on the wave-ui click location dataset have been created.

## Files Created

### 1. Data Transformation
- **`scripts/transform_for_training.py`**
  - Reads wave-ui/prompts.jsonl
  - Samples click locations within bounding boxes
  - Normalizes coordinates to 1400x800
  - Outputs LLaMA-Factory conversation format
  - Handles train/validation splitting

### 2. Training Configuration
- **`configs/qwen_vl_lora.yaml`**
  - LoRA configuration (rank=64, alpha=128)
  - Training hyperparameters optimized for 8xH100
  - Batch size: 2 per device × 4 grad accum × 8 GPUs = 64 effective
  - Learning rate: 5e-5 with cosine schedule
  - 2.5 epochs (~6-7 hours on 8xH100)

- **`configs/deepspeed_zero2.json`**
  - DeepSpeed ZeRO-2 optimization
  - BF16 precision for H100
  - Optimized for multi-GPU training

- **`configs/dataset_info.json`**
  - Dataset registration for LLaMA-Factory
  - Defines train and validation sets

### 3. Training Script
- **`scripts/train.py`**
  - Main training launcher
  - Environment setup
  - Dependency checking
  - Support for single or multi-GPU training
  - DeepSpeed integration
  - Resume from checkpoint capability

### 4. SLURM Job Scripts
- **`slurm/prepare_data.slurm`**
  - Data preprocessing job (1 hour, CPU only)
  - Transforms dataset
  - Validates output
  - Creates train/val splits

- **`slurm/train_qwen_vl.slurm`**
  - Main training job (8 hours, 8xH100)
  - Environment setup
  - Multi-GPU training with DeepSpeed
  - Monitoring and logging
  - Automatic validation

### 5. Setup and Documentation
- **`setup_training.sh`**
  - Automated setup script
  - Creates virtual environment
  - Installs all dependencies
  - Clones LLaMA-Factory
  - Verifies installation

- **`requirements-training.txt`**
  - All Python dependencies
  - PyTorch, Transformers, DeepSpeed
  - Flash Attention 2
  - TensorBoard, Weights & Biases

- **`TRAINING_README.md`**
  - Comprehensive documentation
  - Setup instructions
  - Usage examples
  - Configuration details
  - Troubleshooting guide
  - Inference examples

- **`SETUP_SUMMARY.md`** (this file)
  - Overview of all created files
  - Quick start guide

## Quick Start

### Option 1: Automated Setup
```bash
chmod +x setup_training.sh
./setup_training.sh
```

### Option 2: Manual Setup

#### Step 0: Download Dataset (if needed)
```bash
# Configure and run download
# Edit slurm/download_dataset.slurm first, then:
sbatch slurm/download_dataset.slurm

# Or use Python script directly
python scripts/download_dataset.py --url https://your-url/dataset.tar.gz
```

#### Step 1: Install Dependencies
```bash
pip install -r requirements-training.txt
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e . && cd ..
```

#### Step 2: Prepare Data
```bash
# Local (dataset at /hai/scratch/websight-v2/data)
python scripts/transform_for_training.py

# Or via SLURM
sbatch slurm/prepare_data.slurm
```

> **Dataset Location**: `/hai/scratch/asanshay/websight-v2/data/prompts.jsonl` with images in `/hai/scratch/asanshay/websight-v2/data/images/`

#### Step 3: Train Model
```bash
# Via SLURM (recommended)
sbatch slurm/train_qwen_vl.slurm

# Or local testing
python scripts/train.py --num-gpus 8
```

## Dataset Format

**Input** (`/hai/scratch/asanshay/websight-v2/data/prompts.jsonl`):
```json
{
  "id": 1,
  "original": {
    "bbox": [792.46875, 56.0, 847.46875, 120.0],
    "resolution": [1280, 720],
    "image_path": "images/000000.png",
    ...
  },
  "prompt": "click on the product link"
}
```

**Output** (`data/wave_ui_train.jsonl`):
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a GUI automation assistant. Given an image and a user instruction, output the exact pyautogui.click(x, y) command to execute the action. Coordinates are normalized to 1400x800 resolution."
    },
    {
      "role": "user",
      "content": "<image>\nclick on the product link"
    },
    {
      "role": "assistant",
      "content": "pyautogui.click(892, 336)"
    }
  ],
  "images": ["/hai/scratch/asanshay/websight-v2/data/images/000000.png"]
}
```

## Training Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-VL-8B-Instruct | ~8B parameters |
| Method | LoRA | Rank=64, Alpha=128 |
| GPUs | 8xH100 | 80GB each |
| Batch Size | 64 effective | 2 per device × 4 grad accum × 8 GPUs |
| Learning Rate | 5e-5 | Cosine schedule with 5% warmup |
| Epochs | 2.5 | ~6-7 hours |
| Precision | BF16 | Optimal for H100 |
| Optimizer | AdamW | Weight decay 0.01 |
| Dataset | ~79k samples | ~71k train, ~7.9k val |

## Expected Results

- **Training time**: 6-7 hours on 8xH100
- **Throughput**: 10-15 samples/sec
- **Memory usage**: 60-70GB per GPU
- **Final loss**: Should converge to <0.5
- **Model size**: LoRA adapters ~500MB

## Output

Trained model saved to:
```
saves/qwen3-vl-8b/lora/sft/
├── adapter_config.json
├── adapter_model.safetensors
├── trainer_state.json
└── training_args.bin
```

## Inference Example

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# Load model
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, "saves/qwen3-vl-8b/lora/sft")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Inference
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "path/to/ui.png"},
        {"type": "text", "text": "click on the login button"}
    ]
}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"Command: {result}")  # e.g., "pyautogui.click(945, 523)"

# Execute the click command
exec(result)
```

## Monitoring

- **TensorBoard**: `tensorboard --logdir logs/qwen3-vl-8b`
- **SLURM logs**: `logs/train_qwen3vl_*.out`
- **Checkpoints**: Saved every 500 steps

## Support

For detailed information, see:
- **TRAINING_README.md** - Comprehensive guide
- **configs/qwen_vl_lora.yaml** - Configuration details
- **scripts/** - Source code with comments

For issues with:
- **Qwen3-VL**: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory

## Next Steps

1. ✓ Run setup: `./setup_training.sh`
2. ✓ Prepare data: `sbatch slurm/prepare_data.slurm`
3. ✓ Start training: `sbatch slurm/train_qwen_vl.slurm`
4. ⏳ Monitor progress: `tail -f logs/train_qwen3vl_*.out`
5. ⏳ Evaluate model: Test on validation set
6. ⏳ Deploy: Merge LoRA weights or use adapter directly

Happy training! 🚀

