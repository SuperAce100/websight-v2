# Converting DeepSpeed Checkpoint to Inference-Ready Model

## Problem

You have DeepSpeed checkpoint files (optimizer states and model states) but need to run inference. For LoRA models, you need the adapter weights in PEFT format.

## Solution Options

### Option 1: Check if Adapter Already Exists (Recommended First Step)

LLaMA-Factory typically saves LoRA adapters at the output directory level, not inside checkpoint subdirectories.

Check the parent directory of your checkpoint:
```bash
# Your checkpoint is at:
# ht-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200/global_step200

# Check if adapter exists here:
ls -la ht-v2/saves/qwen3-vl-8b/lora/sft-noeval/

# Look for:
# - adapter_config.json
# - adapter_model.bin (or adapter_model.safetensors)
```

If these files exist, you can use them directly for inference!

### Option 2: Extract Adapter from Checkpoint

If adapter files don't exist, extract them using one of these methods:

#### Method A: Using LLaMA-Factory Export (Recommended)

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path ht-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200 \
    --export_dir ht-v2/saves/qwen3-vl-8b/lora/sft-noeval-adapter \
    --template qwen2_vl \
    --finetuning_type lora \
    --export_size 2 \
    --export_legacy_format False
```

#### Method B: Using Python Script

```bash
python scripts/extract_adapter_from_checkpoint.py \
    --checkpoint_dir ht-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200/global_step200 \
    --output_dir ht-v2/saves/qwen3-vl-8b/lora/sft-noeval-adapter \
    --base_model Qwen/Qwen3-VL-8B-Instruct
```

### Option 3: Load Checkpoint Directly (If LLaMA-Factory Supports It)

Some versions of LLaMA-Factory can load DeepSpeed checkpoints directly. Try:

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path ht-v2/saves/qwen3-vl-8b/lora/sft-noeval \
    --export_dir ht-v2/saves/qwen3-vl-8b/lora/sft-noeval-adapter \
    --template qwen2_vl \
    --finetuning_type lora
```

## Running Inference

Once you have the adapter files, use them for inference:

### Using test_model.py

```bash
python test_model.py \
    --adapter-path ht-v2/saves/qwen3-vl-8b/lora/sft-noeval-adapter \
    --image path/to/image.png \
    --prompt "click the login button"
```

### Using test_after_grounding.py

```bash
python scripts/test_after_grounding.py \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --adapter-path ht-v2/saves/qwen3-vl-8b/lora/sft-noeval-adapter \
    --test-file data/wave_ui_test.jsonl \
    --media-dir data \
    --output inference_results.jsonl
```

### Using Python Code Directly

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "ht-v2/saves/qwen3-vl-8b/lora/sft-noeval-adapter",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model.eval()

# Run inference...
```

## Understanding Checkpoint Structure

- **DeepSpeed Checkpoint**: Contains optimizer states (`bf16_zero_pp_rank_*_optim_states.pt`) and model states (`mp_rank_00_model_states.pt`)
- **LoRA Adapter**: Contains only the adapter weights (`adapter_model.bin`) and config (`adapter_config.json`)

For inference, you only need the adapter files, not the optimizer states.

## Troubleshooting

1. **Adapter files not found**: Check the parent directory of checkpoints
2. **Export fails**: Try using the checkpoint directory (without `global_step200`) as adapter path
3. **Memory issues**: Use `--export_device cpu` in llamafactory-cli export
4. **Wrong format**: Ensure you're using `--finetuning_type lora` in export command

