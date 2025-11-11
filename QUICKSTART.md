# Qwen3-VL Fine-tuning - Quick Start Guide

## TL;DR - Get Training in 4 Steps

### 0️⃣ Download Dataset (if needed, ~1-2 hours)

```bash
# Default: Downloads from HuggingFace (agentsea/wave-ui)
# Automatically extracts parquet files to JSONL + images
./scripts/download_dataset.sh

# Or use Python script directly (same default)
python scripts/download_dataset.py

# Or use SLURM: sbatch slurm/download_dataset.slurm
```

### 1️⃣ Setup (5 minutes)

```bash
# Complete setup: creates venv, installs everything
./setup.sh

# Then activate the environment
source venv/bin/activate
```

### 2️⃣ Prepare Data (~10 minutes)

```bash
# Direct execution (recommended)
./scripts/prepare_data.sh

# Or via SLURM: sbatch slurm/prepare_data.slurm
```

### 3️⃣ Start Training (via SLURM, 8 hours)

```bash
sbatch slurm/train_qwen_vl.slurm
# Monitor: tail -f logs/train_qwen3vl_*.out
```

---

## What This Does

Trains **Qwen3-VL-8B** to generate PyAutoGUI commands from:

- 📸 UI screenshot
- 📝 Natural language instruction (e.g., "click on the login button")
- 📊 Output: PyAutoGUI command (e.g., "pyautogui.click(945, 523)")

**Dataset**: ~79k image-instruction-location pairs from `/hai/scratch/asanshay/websight-v2/data`

---

## CPU Jobs (No SLURM Needed)

All data preparation can run directly as shell scripts:

```bash
# Download dataset
./scripts/download_dataset.sh

# Prepare training data
./scripts/prepare_data.sh
```

**Environment variables for customization:**

```bash
# Custom dataset location for data prep
DATA_DIR=/custom/path ./scripts/prepare_data.sh

# Download options (default: agentsea/wave-ui from HuggingFace)
HF_REPO=agentsea/wave-ui ./scripts/download_dataset.sh     # HuggingFace
DOWNLOAD_URL=https://your-url ./scripts/download_dataset.sh  # From URL
LOCAL_PATH=/existing/dataset ./scripts/download_dataset.sh   # Local copy
```

> **Note**: Dataset defaults to `/hai/scratch/asanshay/websight-v2/data/prompts.jsonl`

---

## Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/qwen3-vl-8b

# SLURM logs
tail -f logs/train_qwen3vl_<job_id>.out

# Check GPU usage
srun --jobid=<job_id> nvidia-smi
```

---

## After Training

**Model location**: `saves/qwen3-vl-8b/lora/sft/`

**Use for inference**:

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load model + LoRA adapter
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", device_map="auto"
)
model = PeftModel.from_pretrained(model, "saves/qwen3-vl-8b/lora/sft")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Predict click location
messages = [{"role": "user", "content": [
    {"type": "image", "image": "screenshot.png"},
    {"type": "text", "text": "click on the submit button"}
]}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
command = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"Command: {command}")  # e.g., "pyautogui.click(789, 456)"

# Execute the command
exec(command)
```

---

## Output Format

**Input**: Natural language + Image

```
System: You are a GUI automation assistant. Given an image and a user instruction,
        output the exact pyautogui.click(x, y) command to execute the action.
        Coordinates are normalized to 1400x800 resolution.

Image: [UI screenshot]
Prompt: "click on the login button"
```

**Output**: PyAutoGUI command (1400x800 normalized)

```python
pyautogui.click(945, 523)
```

The model learns to map visual elements + instructions → executable PyAutoGUI commands.

---

## Files Created

```
✓ setup.sh                           - Complete setup (venv + dependencies)
✓ scripts/download_dataset.sh        - Download dataset (CPU shell script)
✓ scripts/prepare_data.sh            - Prepare data (CPU shell script)
✓ scripts/download_dataset.py        - Dataset download utility
✓ scripts/transform_for_training.py  - Data transformation
✓ scripts/train.py                   - Training launcher
✓ configs/qwen_vl_lora.yaml         - Training config
✓ configs/deepspeed_zero2.json      - DeepSpeed config
✓ configs/dataset_info.json         - Dataset registration
✓ slurm/download_dataset.slurm      - Dataset download job
✓ slurm/prepare_data.slurm          - Data prep job
✓ slurm/train_qwen_vl.slurm         - Training job
✓ requirements-training.txt          - Dependencies
✓ TRAINING_README.md                 - Full documentation
✓ SETUP_SUMMARY.md                   - Detailed overview
✓ DATASET_INFO.md                    - Dataset location & structure
```

---

## Training Specs

| Item    | Value                |
| ------- | -------------------- |
| Model   | Qwen3-VL-8B-Instruct |
| GPUs    | 8xH100 (80GB each)   |
| Time    | ~6-7 hours           |
| Method  | LoRA (rank=64)       |
| Batch   | 64 effective         |
| Dataset | ~79k samples         |

---

## Troubleshooting

| Issue                   | Solution                                       |
| ----------------------- | ---------------------------------------------- |
| OOM error               | Reduce `per_device_train_batch_size` in config |
| SLURM job pending       | Check partition name, adjust in `.slurm` files |
| LLaMA-Factory not found | Run `./setup_training.sh`                      |
| Slow training           | Verify Flash Attention 2 installed             |
| Data not found          | Run `sbatch slurm/prepare_data.slurm` first    |

---

## Need More Details?

- **Full docs**: `TRAINING_README.md`
- **Setup overview**: `SETUP_SUMMARY.md`
- **Dataset download**: `DOWNLOAD_GUIDE.md`
- **Dataset info**: `DATASET_INFO.md`
- **Config details**: `configs/qwen_vl_lora.yaml`

---

**Ready? Run:** `./setup_training.sh` 🚀
