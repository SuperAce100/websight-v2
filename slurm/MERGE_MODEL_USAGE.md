# Using the Merge Model SLURM Script

## Quick Start

### Option 1: Just Merge (For Download)

```bash
# On the cluster
sbatch --account=ingrai slurm/merge_model.slurm
```

This will:
- Merge the adapter from `saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200`
- Save merged model to `saves/qwen3-vl-8b-merged`
- **NOT** push to HuggingFace (you can download it manually)

### Option 2: Merge with Custom Paths

```bash
# Set custom paths
export ADAPTER_PATH="saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200"
export OUTPUT_DIR="saves/qwen3-vl-8b-merged-custom"

# Submit job
sbatch --account=ingrai slurm/merge_model.slurm
```

### Option 3: Merge and Push to HuggingFace

```bash
# Set HuggingFace credentials
export HF_USERNAME="your-username"
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export HUB_MODEL_ID="your-username/qwen3-vl-8b-websight"  # Optional
export PUSH_TO_HUB="true"

# Optional: Make repository private
export PRIVATE_REPO="true"  # or "false"

# Submit job
sbatch --account=ingrai slurm/merge_model.slurm
```

## Default Configuration

The script uses these defaults (can be overridden with environment variables):

- **Base Model**: `Qwen/Qwen3-VL-8B-Instruct`
- **Adapter Path**: `saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200`
- **Output Directory**: `saves/qwen3-vl-8b-merged`
- **Push to Hub**: `false` (set `PUSH_TO_HUB="true"` to enable)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Base model name/path |
| `ADAPTER_PATH` | `saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200` | Path to LoRA adapter |
| `OUTPUT_DIR` | `saves/qwen3-vl-8b-merged` | Output directory for merged model |
| `HF_USERNAME` | (none) | HuggingFace username (required if pushing) |
| `HF_TOKEN` | (none) | HuggingFace token (required if pushing) |
| `HUB_MODEL_ID` | `{HF_USERNAME}/qwen3-vl-8b-websight-merged` | HuggingFace repo ID |
| `PUSH_TO_HUB` | `false` | Set to `"true"` to push to HuggingFace |
| `PRIVATE_REPO` | `false` | Set to `"true"` for private repository |

## Examples

### Example 1: Basic Merge

```bash
sbatch --account=ingrai slurm/merge_model.slurm
```

### Example 2: Merge Different Checkpoint

```bash
export ADAPTER_PATH="saves/qwen3-vl-8b/lora/sft/checkpoint-500"
export OUTPUT_DIR="saves/qwen3-vl-8b-merged-checkpoint500"
sbatch --account=ingrai slurm/merge_model.slurm
```

### Example 3: Merge and Push Public Repo

```bash
export HF_USERNAME="asanshay"
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export PUSH_TO_HUB="true"
export HUB_MODEL_ID="asanshay/qwen3-vl-8b-websight"
sbatch --account=ingrai slurm/merge_model.slurm
```

### Example 4: Merge and Push Private Repo

```bash
export HF_USERNAME="asanshay"
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export PUSH_TO_HUB="true"
export PRIVATE_REPO="true"
export HUB_MODEL_ID="asanshay/qwen3-vl-8b-websight-private"
sbatch --account=ingrai slurm/merge_model.slurm
```

## Checking Job Status

```bash
# Check job status
squeue -u $USER

# View output (while running or after completion)
tail -f logs/merge_model_<JOB_ID>.out

# View errors
tail -f logs/merge_model_<JOB_ID>.err

# After completion, check the output file
cat logs/merge_model_<JOB_ID>.out
```

## Resource Requirements

The script requests:
- **Time**: 2 hours (should be enough for merging)
- **Memory**: 64GB RAM
- **GPU**: 1 GPU (for faster merging)
- **CPUs**: 8 cores

If you need more time, edit the script:
```bash
#SBATCH --time=04:00:00  # Change to 4 hours
```

## After Merging

### Download to Local Computer

```bash
# From your local computer
rsync -avz --progress \
    asanshay@haic:/hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged/ \
    ~/local/path/qwen3-vl-8b-merged/
```

### Use for Inference

```bash
# On cluster or local
python test_model.py \
    --model-path saves/qwen3-vl-8b-merged \
    --image screenshot.png \
    --prompt "click the login button"
```

## Troubleshooting

### Job Fails with "Adapter not found"
- Check that `ADAPTER_PATH` is correct
- Verify the adapter directory contains `adapter_config.json`

### Out of Memory
- The script uses 64GB RAM by default
- If needed, increase memory: `#SBATCH --mem=128G`

### Merge Takes Too Long
- Normal merge time: 10-30 minutes
- If it takes longer, check GPU availability
- Consider using CPU if GPU is busy: edit script to use `--device cpu`

### HuggingFace Push Fails
- Verify `HF_TOKEN` is correct
- Check internet connectivity from cluster
- Ensure repository name doesn't already exist (or use different name)

## Output Files

After successful merge, you'll find:
- Merged model in `saves/qwen3-vl-8b-merged/` (or your `OUTPUT_DIR`)
- Logs in `logs/merge_model_<JOB_ID>.out` and `.err`
- Model card (README.md) in the output directory

