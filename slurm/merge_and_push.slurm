#!/bin/bash
#SBATCH --job-name=merge_push_qwen3vl
#SBATCH --account=ingrai
#SBATCH --output=logs/merge_push_%j.out
#SBATCH --error=logs/merge_push_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Merge LoRA adapter with base model and push to HuggingFace
# This script:
# 1. Merges the LoRA adapter with Qwen3-VL-8B-Instruct
# 2. Pushes the merged model to HuggingFace Hub
# 3. Optionally pushes the LoRA adapter separately

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Qwen3-VL Merge and Push to HuggingFace"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Configuration
WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

# Model paths
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_PATH="saves/qwen3-vl-8b/lora/sft"
MERGED_MODEL_DIR="saves/qwen3-vl-8b-merged"

# HuggingFace configuration
# Set these environment variables before running:
# export HF_USERNAME="your-username"
# export HF_TOKEN="your-token"
# Or set them here (not recommended for security):
# HF_USERNAME="your-username"
# HF_TOKEN="hf_xxxxxxxxxxxxx"

# Check for HuggingFace credentials
if [ -z "${HF_USERNAME:-}" ]; then
    echo "❌ Error: HF_USERNAME not set!"
    echo "Please set your HuggingFace username:"
    echo "  export HF_USERNAME='your-username'"
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "❌ Error: HF_TOKEN not set!"
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN='hf_xxxxxxxxxxxxx'"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# Repository names on HuggingFace
MERGED_REPO_NAME="${HF_USERNAME}/qwen3-vl-8b-websight-merged"
ADAPTER_REPO_NAME="${HF_USERNAME}/qwen3-vl-8b-websight-lora"

# Options
PUSH_MERGED=true      # Push merged model
PUSH_ADAPTER=true     # Push LoRA adapter separately
PRIVATE_REPO=false    # Make repositories private

echo "Configuration:"
echo "  Base model: ${BASE_MODEL}"
echo "  Adapter path: ${ADAPTER_PATH}"
echo "  Merged model dir: ${MERGED_MODEL_DIR}"
echo "  HuggingFace user: ${HF_USERNAME}"
echo "  Merged repo: ${MERGED_REPO_NAME}"
echo "  Adapter repo: ${ADAPTER_REPO_NAME}"
echo "  Private repos: ${PRIVATE_REPO}"
echo ""

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

# Set environment variables
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1  # Faster uploads
mkdir -p "${HF_HOME}"

# Verify adapter exists
echo "========================================"
echo "Verifying LoRA adapter..."
echo "========================================"

if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "❌ Error: LoRA adapter not found at ${ADAPTER_PATH}"
    echo "Please ensure training completed successfully."
    exit 1
fi

if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
    echo "❌ Error: adapter_config.json not found in ${ADAPTER_PATH}"
    exit 1
fi

echo "✓ LoRA adapter found"
echo ""

# Install/verify dependencies
echo "========================================"
echo "Installing dependencies..."
echo "========================================"

pip install -q --upgrade pip
pip install -q transformers>=4.57.0
pip install -q accelerate
pip install -q peft
pip install -q huggingface_hub
pip install -q hf-transfer  # For faster uploads

echo "✓ Dependencies installed"
echo ""

# Login to HuggingFace
echo "========================================"
echo "Logging in to HuggingFace..."
echo "========================================"

huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

echo "✓ Logged in to HuggingFace"
echo ""

# Merge LoRA adapter with base model
if [ "${PUSH_MERGED}" = true ]; then
    echo "========================================"
    echo "Merging LoRA adapter with base model..."
    echo "========================================"
    
    # Remove old merged model if exists
    if [ -d "${MERGED_MODEL_DIR}" ]; then
        echo "Removing old merged model..."
        rm -rf "${MERGED_MODEL_DIR}"
    fi
    
    # Merge using LLaMA-Factory
    llamafactory-cli export \
        --model_name_or_path "${BASE_MODEL}" \
        --adapter_name_or_path "${ADAPTER_PATH}" \
        --template qwen2_vl \
        --finetuning_type lora \
        --export_dir "${MERGED_MODEL_DIR}" \
        --export_size 2 \
        --export_legacy_format False
    
    if [ $? -eq 0 ]; then
        echo "✓ Model merged successfully"
        echo "  Location: ${MERGED_MODEL_DIR}"
        
        # Check merged model size
        MERGED_SIZE=$(du -sh "${MERGED_MODEL_DIR}" | cut -f1)
        echo "  Size: ${MERGED_SIZE}"
    else
        echo "❌ Error: Model merge failed"
        exit 1
    fi
    echo ""
    
    # Create model card for merged model
    echo "Creating model card for merged model..."
    cat > "${MERGED_MODEL_DIR}/README.md" << 'EOF'
---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
tags:
  - qwen3-vl
  - vision
  - gui-automation
  - websight
  - fine-tuned
datasets:
  - wave-ui/websight-v2
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3-VL-8B WebSight Fine-tuned

This model is a fine-tuned version of [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) on the WebSight dataset for GUI automation tasks.

## Model Description

- **Base Model**: Qwen3-VL-8B-Instruct
- **Fine-tuning Method**: LoRA (merged)
- **Dataset**: wave-ui/websight-v2
- **Task**: Image-to-click location prediction
- **Output Format**: `pyautogui.click(x, y)` commands

## Usage

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    "REPO_NAME",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("REPO_NAME")

# Prepare input
image = Image.open("screenshot.png")
prompt = "click the login button"

inputs = processor(
    text=f"<image>\n{prompt}",
    images=image,
    return_tensors="pt"
).to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)  # Output: pyautogui.click(x, y)
```

## Training Details

- **Training Framework**: LLaMA-Factory
- **Hardware**: 8x H100 GPUs
- **Training Time**: ~8 hours
- **LoRA Config**:
  - Rank: 64
  - Alpha: 128
  - Dropout: 0.05
  - Target modules: q_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Output Format

The model outputs click coordinates normalized to 1400x800 resolution:
- Format: `pyautogui.click(x, y)`
- Example: `pyautogui.click(565, 486)`

Scale to your screen resolution:
```python
x_actual = int(x_norm * (screen_width / 1400))
y_actual = int(y_norm * (screen_height / 800))
```

## Citation

```bibtex
@misc{qwen3-vl-websight,
  title={Qwen3-VL Fine-tuned for GUI Automation},
  author={Your Name},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/REPO_NAME}}
}
```

## License

Apache 2.0 (inherited from base model)
EOF
    
    # Replace REPO_NAME in README
    sed -i "s|REPO_NAME|${MERGED_REPO_NAME}|g" "${MERGED_MODEL_DIR}/README.md" 2>/dev/null || \
        sed -i '' "s|REPO_NAME|${MERGED_REPO_NAME}|g" "${MERGED_MODEL_DIR}/README.md"
    
    echo "✓ Model card created"
    echo ""
    
    # Push merged model to HuggingFace
    echo "========================================"
    echo "Pushing merged model to HuggingFace..."
    echo "========================================"
    echo "Repository: ${MERGED_REPO_NAME}"
    
    # Create repository
    if [ "${PRIVATE_REPO}" = true ]; then
        huggingface-cli repo create "${MERGED_REPO_NAME}" --type model --private || true
    else
        huggingface-cli repo create "${MERGED_REPO_NAME}" --type model || true
    fi
    
    # Push using Python script for better control
    python3 << PYPUSH
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = "${MERGED_REPO_NAME}"
folder_path = "${MERGED_MODEL_DIR}"

print(f"Uploading merged model to {repo_id}...")
print(f"This may take 10-30 minutes depending on your connection...")

try:
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned Qwen3-VL-8B model (merged)",
    )
    print(f"✓ Merged model uploaded successfully!")
    print(f"  View at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"❌ Error uploading merged model: {e}")
    exit(1)
PYPUSH
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Merged model pushed to HuggingFace"
        echo "  URL: https://huggingface.co/${MERGED_REPO_NAME}"
    else
        echo "❌ Error: Failed to push merged model"
        exit 1
    fi
    echo ""
fi

# Push LoRA adapter separately
if [ "${PUSH_ADAPTER}" = true ]; then
    echo "========================================"
    echo "Pushing LoRA adapter to HuggingFace..."
    echo "========================================"
    echo "Repository: ${ADAPTER_REPO_NAME}"
    
    # Create model card for adapter
    cat > "${ADAPTER_PATH}/README.md" << 'EOF'
---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
tags:
  - qwen3-vl
  - vision
  - gui-automation
  - websight
  - lora
  - peft
datasets:
  - wave-ui/websight-v2
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3-VL-8B WebSight LoRA Adapter

This is a LoRA adapter for [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) fine-tuned on the WebSight dataset for GUI automation tasks.

## Model Description

- **Base Model**: Qwen3-VL-8B-Instruct
- **Adapter Type**: LoRA
- **Dataset**: wave-ui/websight-v2
- **Task**: Image-to-click location prediction
- **Output Format**: `pyautogui.click(x, y)` commands

## Usage

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
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
    "ADAPTER_REPO_NAME",
    device_map="auto"
)

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Prepare input
image = Image.open("screenshot.png")
prompt = "click the login button"

inputs = processor(
    text=f"<image>\n{prompt}",
    images=image,
    return_tensors="pt"
).to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)  # Output: pyautogui.click(x, y)
```

## LoRA Configuration

- **Rank**: 64
- **Alpha**: 128
- **Dropout**: 0.05
- **Target Modules**: q_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Training Details

- **Training Framework**: LLaMA-Factory
- **Hardware**: 8x H100 GPUs
- **Training Time**: ~8 hours
- **Batch Size**: 2 per device × 4 grad accum × 8 GPUs = 64 effective
- **Learning Rate**: 5e-5
- **Epochs**: 2.5

## Advantages of LoRA

- **Smaller Size**: ~100-500 MB vs. 16 GB for full model
- **Faster Loading**: Load only the adapter weights
- **Easy Switching**: Swap adapters on the same base model
- **Storage Efficient**: Share base model across multiple adapters

## Output Format

The model outputs click coordinates normalized to 1400x800 resolution:
- Format: `pyautogui.click(x, y)`
- Example: `pyautogui.click(565, 486)`

Scale to your screen resolution:
```python
x_actual = int(x_norm * (screen_width / 1400))
y_actual = int(y_norm * (screen_height / 800))
```

## Citation

```bibtex
@misc{qwen3-vl-websight-lora,
  title={Qwen3-VL LoRA Adapter for GUI Automation},
  author={Your Name},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/ADAPTER_REPO_NAME}}
}
```

## License

Apache 2.0 (inherited from base model)
EOF
    
    # Replace ADAPTER_REPO_NAME in README
    sed -i "s|ADAPTER_REPO_NAME|${ADAPTER_REPO_NAME}|g" "${ADAPTER_PATH}/README.md" 2>/dev/null || \
        sed -i '' "s|ADAPTER_REPO_NAME|${ADAPTER_REPO_NAME}|g" "${ADAPTER_PATH}/README.md"
    
    echo "✓ Adapter model card created"
    echo ""
    
    # Create repository
    if [ "${PRIVATE_REPO}" = true ]; then
        huggingface-cli repo create "${ADAPTER_REPO_NAME}" --type model --private || true
    else
        huggingface-cli repo create "${ADAPTER_REPO_NAME}" --type model || true
    fi
    
    # Push adapter
    python3 << PYPUSH
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = "${ADAPTER_REPO_NAME}"
folder_path = "${ADAPTER_PATH}"

print(f"Uploading LoRA adapter to {repo_id}...")
print(f"This should be quick (adapter is small)...")

try:
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload LoRA adapter for Qwen3-VL-8B",
    )
    print(f"✓ LoRA adapter uploaded successfully!")
    print(f"  View at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"❌ Error uploading adapter: {e}")
    exit(1)
PYPUSH
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ LoRA adapter pushed to HuggingFace"
        echo "  URL: https://huggingface.co/${ADAPTER_REPO_NAME}"
    else
        echo "❌ Error: Failed to push LoRA adapter"
        exit 1
    fi
    echo ""
fi

# Summary
echo "========================================"
echo "✓ All tasks completed successfully!"
echo "========================================"
echo ""

if [ "${PUSH_MERGED}" = true ]; then
    echo "📦 Merged Model:"
    echo "   https://huggingface.co/${MERGED_REPO_NAME}"
    echo ""
fi

if [ "${PUSH_ADAPTER}" = true ]; then
    echo "🔌 LoRA Adapter:"
    echo "   https://huggingface.co/${ADAPTER_REPO_NAME}"
    echo ""
fi

echo "Usage examples:"
echo ""
echo "# Use merged model:"
echo "from transformers import AutoModelForVision2Seq"
echo "model = AutoModelForVision2Seq.from_pretrained('${MERGED_REPO_NAME}')"
echo ""
echo "# Use LoRA adapter:"
echo "from peft import PeftModel"
echo "model = PeftModel.from_pretrained(base_model, '${ADAPTER_REPO_NAME}')"
echo ""
echo "End time: $(date)"
echo "========================================"

