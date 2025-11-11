# Qwen3-VL Fine-tuning - Quick Start Guide

## TL;DR - Get Training in 3 Steps

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

Trains **Qwen3-VL-8B** to predict click locations from:
- 📸 UI screenshot
- 📝 Natural language instruction (e.g., "click on the login button")
- 📊 Output: normalized coordinates (e.g., "945, 523")

**Dataset**: ~79k image-instruction-location pairs from wave-ui

---

## Alternative: Local Testing

```bash
# Prepare data locally
python scripts/transform_for_training.py

# Test on 1 GPU (for debugging)
python scripts/train.py --num-gpus 1 --no-deepspeed
```

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
coords = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"Click at: {coords}")  # e.g., "789, 456"
```

---

## Files Created

```
✓ scripts/transform_for_training.py  - Data transformation
✓ scripts/train.py                   - Training launcher
✓ configs/qwen_vl_lora.yaml         - Training config
✓ configs/deepspeed_zero2.json      - DeepSpeed config
✓ configs/dataset_info.json         - Dataset registration
✓ slurm/prepare_data.slurm          - Data prep job
✓ slurm/train_qwen_vl.slurm         - Training job
✓ setup_training.sh                  - Auto-setup script
✓ requirements-training.txt          - Dependencies
✓ TRAINING_README.md                 - Full documentation
✓ SETUP_SUMMARY.md                   - Detailed overview
```

---

## Training Specs

| Item | Value |
|------|-------|
| Model | Qwen3-VL-8B-Instruct |
| GPUs | 8xH100 (80GB each) |
| Time | ~6-7 hours |
| Method | LoRA (rank=64) |
| Batch | 64 effective |
| Dataset | ~79k samples |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM error | Reduce `per_device_train_batch_size` in config |
| SLURM job pending | Check partition name, adjust in `.slurm` files |
| LLaMA-Factory not found | Run `./setup_training.sh` |
| Slow training | Verify Flash Attention 2 installed |
| Data not found | Run `sbatch slurm/prepare_data.slurm` first |

---

## Need More Details?

- **Full docs**: `TRAINING_README.md`
- **Setup overview**: `SETUP_SUMMARY.md`
- **Config details**: `configs/qwen_vl_lora.yaml`

---

**Ready? Run:** `./setup_training.sh` 🚀

