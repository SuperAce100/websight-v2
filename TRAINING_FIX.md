# Training Script Fix - November 11, 2025

## Problem

The training job was starting but all processes were being killed after ~25 seconds with error:
```
[ERROR] [launch.py:341:sigkill_handler] [...] exits with return code = 1
```

## Root Cause

The SLURM script was incorrectly calling LLaMA-Factory:
```bash
# WRONG - doesn't work this way
deepspeed --num_gpus=8 --master_port=29500 \
    LLaMA-Factory/src/train.py \
    configs/qwen_vl_lora.yaml \  # YAML not parsed correctly
    --dataset_dir ...
```

## Solution

### 1. Fixed Training Command

Changed to use `llamafactory-cli` with explicit command-line arguments:

```bash
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --stage sft \
    --do_train true \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 64 \
    # ... all other parameters explicitly specified
```

### 2. Fixed Dataset Paths

**Before** (`configs/dataset_info.json`):
```json
{
  "wave_ui_dataset": {
    "file_name": "data/wave_ui_train.jsonl",  // WRONG - double "data/"
    ...
  }
}
```

**After**:
```json
{
  "wave_ui_dataset": {
    "file_name": "wave_ui_train.jsonl",  // CORRECT
    ...
  }
}
```

With `--dataset_dir /path/to/data`, this correctly resolves to `/path/to/data/wave_ui_train.jsonl`.

### 3. Added Validation Dataset

Added explicit validation dataset parameter:
```bash
--eval_dataset wave_ui_val \
```

### 4. Fixed Triton Cache Warning

Added environment variable to avoid NFS warnings:
```bash
export TRITON_CACHE_DIR="${WORKSPACE_DIR}/.cache/triton"
mkdir -p "${TRITON_CACHE_DIR}"
```

### 5. Linked Dataset Info

Ensured LLaMA-Factory can find dataset_info.json:
```bash
ln -sf "${WORKSPACE_DIR}/configs/dataset_info.json" LLaMA-Factory/data/dataset_info.json
```

## Files Changed

1. **`slurm/train_qwen_vl.slurm`**
   - Changed from `deepspeed + train.py + YAML` to `llamafactory-cli train` with CLI args
   - Added TRITON_CACHE_DIR setup
   - Added dataset_info.json symlink
   - Added validation dataset

2. **`configs/dataset_info.json`**
   - Fixed file paths: removed "data/" prefix from filenames

## How to Use

### Cancel the old job (if still running)
```bash
scancel <job_id>
```

### Resubmit with fixed script
```bash
sbatch --account=ingrai slurm/train_qwen_vl.slurm
```

### Monitor
```bash
# Check job status
squeue -u $USER

# Watch output log
tail -f logs/train_qwen3vl_*.out

# Watch error log
tail -f logs/train_qwen3vl_*.err

# Monitor GPU usage (if you have interactive access)
watch nvidia-smi
```

## Expected Behavior

After the fix, you should see:
1. Job stays in `squeue` (not disappearing immediately)
2. Log file shows model loading progress
3. Training begins with loss values being logged every 10 steps
4. Evaluation runs every 500 steps

## Verification Checklist

Before resubmitting, verify:
- [ ] `data/wave_ui_train.jsonl` exists
- [ ] `data/wave_ui_val.jsonl` exists
- [ ] `configs/dataset_info.json` has correct paths (no "data/" prefix)
- [ ] `LLaMA-Factory/` directory exists
- [ ] `venv/` is activated in the SLURM script
- [ ] `logs/` directory exists

## Typical Training Output

You should see something like:
```
Loading checkpoint shards: 100%|██████████| 4/4 [00:10<00:00,  2.60s/it]
trainable params: 33,554,432 || all params: 8,033,554,432 || trainable%: 0.4177
***** Running training *****
  Num examples = 71471
  Num Epochs = 2.5
  Instantaneous batch size per device = 2
  Gradient accumulation steps = 4
  Total train batch size = 64
  Total optimization steps = 2793
...
{'loss': 0.5234, 'learning_rate': 4.5e-05, 'epoch': 0.01}
{'loss': 0.4892, 'learning_rate': 4.2e-05, 'epoch': 0.02}
...
```

## Troubleshooting

### If job still fails immediately
```bash
# Check the actual error
tail -50 logs/train_qwen3vl_*.err

# Verify dataset loading works
head -1 data/wave_ui_train.jsonl | python -m json.tool
```

### If "dataset not found" error
```bash
# Verify paths
ls -lh data/wave_ui_train.jsonl
cat configs/dataset_info.json
```

### If OOM (Out of Memory) error
Reduce batch size in the SLURM script:
```bash
--per_device_train_batch_size 1 \  # Instead of 2
--gradient_accumulation_steps 8 \  # Instead of 4
```

## Support

- Full guide: `TRAINING_README.md`
- HAI cluster specifics: `HAI_CLUSTER_GUIDE.md`
- Quick start: `QUICKSTART.md`

