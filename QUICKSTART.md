# Qwen3-VL Fine-tuning - Quick Start Guide

## TL;DR - Get Training in 4 Steps

### 0️⃣ Download Dataset (if needed, ~1-2 hours)

```bash
# Edit slurm/download_dataset.slurm to set your data source, then:
sbatch slurm/download_dataset.slurm
# Or download directly: python scripts/download_dataset.py --url https://your-dataset-url
```

### 1️⃣ Setup (5 minutes)

```bash
./setup_training.sh
```

### 2️⃣ Prepare Data (via SLURM, ~10 minutes)

```bash
sbatch slurm/prepare_data.slurm
# Check status: squeue -u $USER
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

**Dataset**: ~79k image-instruction-location pairs from `/hai/scratch/websight-v2/data`

---

## Alternative: Local Testing

```bash
# Prepare data locally (reads from /hai/scratch/websight-v2/data)
python scripts/transform_for_training.py

# Test on 1 GPU (for debugging)
python scripts/train.py --num-gpus 1 --no-deepspeed
```

> **Note**: Dataset is at `/hai/scratch/websight-v2/data/prompts.jsonl` with images in `images/` subdirectory.

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
✓ scripts/download_dataset.py        - Dataset download utility
✓ scripts/transform_for_training.py  - Data transformation
✓ scripts/train.py                   - Training launcher
✓ configs/qwen_vl_lora.yaml         - Training config
✓ configs/deepspeed_zero2.json      - DeepSpeed config
✓ configs/dataset_info.json         - Dataset registration
✓ slurm/download_dataset.slurm      - Dataset download job
✓ slurm/prepare_data.slurm          - Data prep job
✓ slurm/train_qwen_vl.slurm         - Training job
✓ setup_training.sh                  - Auto-setup script
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
