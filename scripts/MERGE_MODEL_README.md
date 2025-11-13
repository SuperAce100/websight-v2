# Merge LoRA Adapter with Base Model

This script merges your LoRA adapter with the base model, creating a standalone model that can be downloaded and pushed to HuggingFace.

## Quick Start

### Option 1: Simple Bash Script (Easiest)

```bash
# On the cluster
./scripts/merge_model.sh
```

Or with custom paths:
```bash
./scripts/merge_model.sh \
    /path/to/checkpoint-200 \
    /path/to/output/merged-model
```

### Option 2: Python Script (More Control)

```bash
# Just merge (for download)
python scripts/merge_model.py \
    --adapter_path /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200 \
    --output_dir /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged \
    --base_model Qwen/Qwen3-VL-8B-Instruct
```

### Option 3: Merge and Push to HuggingFace

```bash
# Set your HuggingFace token
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Merge and push
python scripts/merge_model.py \
    --adapter_path /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200 \
    --output_dir /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged \
    --base_model Qwen/Qwen3-VL-8B-Instruct \
    --push_to_hub \
    --hub_model_id your-username/qwen3-vl-8b-websight \
    --hub_token $HF_TOKEN
```

## Download Merged Model to Local Computer

After merging on the cluster, download it:

```bash
# From your local computer
rsync -avz --progress \
    asanshay@haic:/hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged/ \
    ~/local/path/qwen3-vl-8b-merged/
```

Or use `scp`:
```bash
scp -r asanshay@haic:/hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged/ \
    ~/local/path/
```

## Push to HuggingFace from Local Computer

After downloading, you can push from your local machine:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Login
huggingface-cli login

# Push the model
cd ~/local/path/qwen3-vl-8b-merged
git lfs install
git init
git add .
git commit -m "Upload merged Qwen3-VL-8B WebSight model"
git remote add origin https://huggingface.co/your-username/qwen3-vl-8b-websight
git push -u origin main
```

Or use the Python API:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="~/local/path/qwen3-vl-8b-merged",
    repo_id="your-username/qwen3-vl-8b-websight",
    repo_type="model"
)
```

## Using the Merged Model

### With test_model.py

```bash
# Use merged model directly (no base model needed)
python test_model.py \
    --model-path /path/to/merged-model \
    --image screenshot.png \
    --prompt "click the login button"
```

### In Python Code

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image

# Load merged model (no adapter needed!)
model = AutoModelForVision2Seq.from_pretrained(
    "/path/to/merged-model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "/path/to/merged-model",
    trust_remote_code=True
)

model.eval()

# Run inference
image = Image.open("screenshot.png")
prompt = "click the login button"

inputs = processor(
    text=f"<image>\n{prompt}",
    images=image,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)  # pyautogui.click(x, y)
```

## Advantages of Merged Model

1. **Standalone**: No need to load base model + adapter separately
2. **Faster Loading**: Single model load instead of two
3. **Easier Distribution**: One directory to share/download
4. **HuggingFace Compatible**: Can be pushed directly to Hub
5. **Simpler Inference**: No PEFT library needed

## Disadvantages

1. **Larger Size**: ~16GB vs ~500MB for adapter
2. **Less Flexible**: Can't swap adapters easily
3. **Takes Longer to Merge**: 10-30 minutes vs instant for adapter

## Troubleshooting

### "llamafactory-cli not found"
The script will fall back to Python API method (slower but works).

### "Out of memory"
Use `--device cpu` to merge on CPU (slower but uses less GPU memory).

### "HuggingFace token not found"
Set `HF_TOKEN` environment variable or use `--hub_token`.

### Merge takes too long
- Using `llamafactory-cli` is faster (default)
- Python API method is slower but more compatible
- Use `--no_llamafactory` to force Python API if needed

## File Structure After Merge

```
merged-model/
├── README.md                    # Model card
├── config.json                  # Model config
├── generation_config.json       # Generation config
├── model-00001-of-00002.safetensors  # Model weights (sharded)
├── model-00002-of-00002.safetensors
├── tokenizer.json              # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
├── merges.txt
└── ... (other tokenizer files)
```

## Next Steps

1. ✅ Merge the model
2. ✅ Download to local computer
3. ✅ Push to HuggingFace (optional)
4. ✅ Use for inference!

