# Qwen3-VL Fine-tuning for Click Location Prediction

This directory contains scripts and configurations to fine-tune **Qwen3-VL-8B-Instruct** on the wave-ui dataset for predicting click locations from natural language instructions.

## Overview

The model learns to predict click coordinates (normalized to 1400x800) given:
- An image of a UI element
- A natural language instruction (e.g., "click on the product link")

## Directory Structure

```
websight-v2/
├── scripts/
│   ├── transform_for_training.py  # Data transformation script
│   └── train.py                    # Main training script
├── configs/
│   ├── qwen_vl_lora.yaml          # Training configuration
│   ├── deepspeed_zero2.json       # DeepSpeed config for distributed training
│   └── dataset_info.json          # Dataset registration for LLaMA-Factory
├── slurm/
│   ├── prepare_data.slurm         # SLURM job for data preparation
│   └── train_qwen_vl.slurm        # SLURM job for training
├── data/                           # Generated training data (created by scripts)
│   ├── wave_ui_train.jsonl
│   └── wave_ui_val.jsonl
├── wave-ui/                        # Source dataset
│   ├── prompts.jsonl
│   └── images/
└── saves/                          # Training checkpoints and outputs
    └── qwen3-vl-8b/lora/sft/
```

## Setup

### 1. Install Dependencies

Using `uv` (recommended):
```bash
uv pip install -r requirements-training.txt
```

Or using `pip`:
```bash
pip install -r requirements-training.txt
```

### 2. Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
cd ..
```

### 3. Install Latest Transformers (for Qwen3-VL support)

```bash
pip install git+https://github.com/huggingface/transformers
```

## Usage

### Step 1: Prepare the Dataset

Transform the wave-ui dataset into the training format:

**Local execution:**
```bash
python scripts/transform_for_training.py \
    --input wave-ui/prompts.jsonl \
    --output-dir data \
    --base-image-path wave-ui \
    --val-ratio 0.1 \
    --seed 42
```

**SLURM execution:**
```bash
sbatch slurm/prepare_data.slurm
```

This will create:
- `data/wave_ui_train.jsonl` - Training set (~71k samples)
- `data/wave_ui_val.jsonl` - Validation set (~7.9k samples)

### Step 2: Launch Training

**SLURM execution (recommended for multi-GPU):**
```bash
sbatch slurm/train_qwen_vl.slurm
```

**Local execution (for testing):**
```bash
# Single GPU
python scripts/train.py --num-gpus 1 --no-deepspeed

# Multi-GPU with DeepSpeed
python scripts/train.py --num-gpus 8
```

### Step 3: Monitor Training

Training logs and metrics:
- **TensorBoard**: `tensorboard --logdir logs/qwen3-vl-8b`
- **SLURM logs**: `logs/train_qwen3vl_<job_id>.out`

Expected training metrics:
- **Duration**: ~6-7 hours on 8xH100 for 2.5 epochs
- **Throughput**: ~10-15 samples/sec
- **Memory usage**: ~60-70GB per GPU with gradient checkpointing
- **Final loss**: Should converge to <0.5 for good performance

## Configuration Details

### Training Hyperparameters

From `configs/qwen_vl_lora.yaml`:

- **Model**: Qwen/Qwen3-VL-8B-Instruct
- **Method**: LoRA fine-tuning
  - Rank: 64
  - Alpha: 128
  - Dropout: 0.05
- **Batch size**: 2 per device × 4 grad accumulation × 8 GPUs = 64 effective
- **Learning rate**: 5e-5 with cosine schedule
- **Epochs**: 2.5
- **Precision**: BF16
- **Optimization**: AdamW with weight decay 0.01

### Hardware Requirements

Recommended: **8xH100 GPUs with 80GB VRAM each**

Minimum requirements:
- 4xA100 (40GB) or better
- ~300GB total GPU memory
- High-speed interconnect (NVLink/InfiniBand)

For smaller setups:
- Reduce batch size in `configs/qwen_vl_lora.yaml`
- Increase gradient accumulation steps
- Consider reducing LoRA rank to 32

## Output Format

The model outputs PyAutoGUI commands:
```
System: You are a GUI automation assistant. Given an image and a user instruction, 
        output the exact pyautogui.click(x, y) command to execute the action. 
        Coordinates are normalized to 1400x800 resolution.

Input: <image>\nclick on the product link
Output: pyautogui.click(892, 336)
```

Coordinates are normalized to 1400x800 resolution and formatted as executable PyAutoGUI commands.

## Inference

After training, use the fine-tuned model:

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# Load base model and LoRA adapter
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
from peft import PeftModel
model = PeftModel.from_pretrained(model, "saves/qwen3-vl-8b/lora/sft")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Inference
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.png"},
            {"type": "text", "text": "click on the login button"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"Command: {result}")  # e.g., "pyautogui.click(892, 336)"

# To execute the command:
# exec(result)
```

## Merging LoRA Weights

To create a standalone model with merged weights:

```bash
python LLaMA-Factory/src/export_model.py \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path saves/qwen3-vl-8b/lora/sft \
    --export_dir saves/qwen3-vl-8b-merged \
    --export_size 2 \
    --export_device cpu
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing (already enabled)
- Reduce LoRA rank to 32

### Slow Training

- Verify Flash Attention 2 is installed: `pip install flash-attn --no-build-isolation`
- Check GPU utilization: `nvidia-smi dmon`
- Ensure high-speed interconnect is working (check NCCL logs)
- Verify data loading isn't a bottleneck (increase `preprocessing_num_workers`)

### Model Not Converging

- Check data quality: `head -n 5 data/wave_ui_train.jsonl`
- Verify image paths are correct
- Try reducing learning rate to 2e-5
- Increase training epochs

### SLURM Job Fails

- Check module availability: `module avail cuda`
- Verify GPU allocation: `squeue -u $USER`
- Check job output: `cat logs/train_qwen3vl_<job_id>.err`
- Adjust partition name in SLURM script based on your cluster

## Citation

If you use this code or the Qwen3-VL model, please cite:

```bibtex
@misc{qwen3technicalreport,
    title={Qwen3 Technical Report}, 
    author={Qwen Team},
    year={2025},
    eprint={2505.09388},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.09388}
}
```

## License

This project follows the license of the underlying models:
- Qwen3-VL: Apache 2.0
- LLaMA-Factory: Apache 2.0

