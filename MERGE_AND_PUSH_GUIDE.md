# Merge and Push to HuggingFace Guide

This guide explains how to merge your fine-tuned LoRA adapter with the base model and push it to HuggingFace Hub.

## Prerequisites

### 1. Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "qwen3-vl-upload")
4. Select "Write" permission
5. Copy the token (starts with `hf_...`)

### 2. Set Environment Variables

On the HAI cluster, set your credentials:

```bash
export HF_USERNAME="your-huggingface-username"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

**Important**: Add these to your `~/.bashrc` to persist across sessions:

```bash
echo 'export HF_USERNAME="your-username"' >> ~/.bashrc
echo 'export HF_TOKEN="hf_xxxxx"' >> ~/.bashrc
source ~/.bashrc
```

## Quick Start

### Submit the Job

```bash
# Make sure you're in the project directory
cd ~/websight-v2

# Submit the merge and push job
sbatch --account=ingrai slurm/merge_and_push.slurm
```

### Monitor Progress

```bash
# Watch the job status
squeue -u $USER

# Follow the log in real-time
tail -f logs/merge_push_<job_id>.out
```

## What the Script Does

The script performs the following steps:

1. **Verifies LoRA adapter** exists at `saves/qwen3-vl-8b/lora/sft`
2. **Merges LoRA with base model** using LLaMA-Factory
3. **Creates model cards** with usage instructions
4. **Pushes merged model** to HuggingFace (repo: `<username>/qwen3-vl-8b-websight-merged`)
5. **Pushes LoRA adapter** separately (repo: `<username>/qwen3-vl-8b-websight-lora`)

## Configuration Options

Edit `slurm/merge_and_push.slurm` to customize:

```bash
# Repository names
MERGED_REPO_NAME="${HF_USERNAME}/qwen3-vl-8b-websight-merged"
ADAPTER_REPO_NAME="${HF_USERNAME}/qwen3-vl-8b-websight-lora"

# What to push
PUSH_MERGED=true      # Push merged model (16GB)
PUSH_ADAPTER=true     # Push LoRA adapter (~500MB)

# Privacy
PRIVATE_REPO=false    # Set to true for private repos
```

## Expected Output

### Merged Model Repository

- **Size**: ~16 GB
- **URL**: `https://huggingface.co/<username>/qwen3-vl-8b-websight-merged`
- **Contents**:
  - Model weights (safetensors)
  - Configuration files
  - Tokenizer
  - README with usage instructions

### LoRA Adapter Repository

- **Size**: ~100-500 MB
- **URL**: `https://huggingface.co/<username>/qwen3-vl-8b-websight-lora`
- **Contents**:
  - LoRA adapter weights
  - Adapter configuration
  - README with usage instructions

## Usage After Upload

### Using the Merged Model

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

model = AutoModelForVision2Seq.from_pretrained(
    "your-username/qwen3-vl-8b-websight-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("your-username/qwen3-vl-8b-websight-merged")
```

### Using the LoRA Adapter

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

# Load your adapter
model = PeftModel.from_pretrained(
    base_model,
    "your-username/qwen3-vl-8b-websight-lora"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
```

## Troubleshooting

### Error: HF_USERNAME not set

```bash
export HF_USERNAME="your-username"
```

### Error: HF_TOKEN not set

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

Get your token from: https://huggingface.co/settings/tokens

### Error: LoRA adapter not found

Make sure training completed successfully:

```bash
ls -lh saves/qwen3-vl-8b/lora/sft/
```

You should see:
- `adapter_config.json`
- `adapter_model.safetensors` (or `.bin`)

### Upload is slow

The script uses `hf-transfer` for faster uploads. If it's still slow:

1. Check your network connection
2. Consider uploading only the LoRA adapter first (set `PUSH_MERGED=false`)
3. The merged model is ~16GB and may take 20-30 minutes

### Repository already exists

If you're re-uploading, the script will update the existing repository. To start fresh:

```bash
# Delete the repository on HuggingFace website, then re-run
```

## Time Estimates

- **Merge**: ~5-10 minutes
- **Upload merged model**: ~20-30 minutes (16GB)
- **Upload LoRA adapter**: ~2-5 minutes (~500MB)
- **Total**: ~30-45 minutes

## Security Notes

1. **Never commit your HF_TOKEN** to git
2. Use environment variables or `.bashrc` for credentials
3. Consider using private repositories for sensitive models
4. Tokens can be revoked at https://huggingface.co/settings/tokens

## Next Steps

After uploading:

1. **Test the model** from HuggingFace:
   ```python
   from transformers import pipeline
   pipe = pipeline("image-text-to-text", model="your-username/qwen3-vl-8b-websight-merged")
   ```

2. **Share your model** by updating the README on HuggingFace

3. **Create a model card** with:
   - Training details
   - Performance metrics
   - Example outputs
   - Limitations

4. **Tag your model** appropriately:
   - `qwen3-vl`
   - `vision`
   - `gui-automation`
   - `websight`

## Support

If you encounter issues:

1. Check the SLURM output log: `logs/merge_push_<job_id>.out`
2. Verify your HuggingFace token is valid
3. Ensure you have write permissions to create repositories
4. Check disk space: `df -h ~/websight-v2`

## References

- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [LLaMA-Factory Export Guide](https://github.com/hiyouga/LLaMA-Factory#export-model)

