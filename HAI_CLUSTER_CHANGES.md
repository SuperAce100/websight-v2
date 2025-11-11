# HAI Cluster Configuration Changes

This document summarizes the changes made to configure the training pipeline for the Stanford HAI Compute Cluster.

## What Changed

### 1. SLURM Job Scripts

Both training SLURM scripts have been updated to work with HAI cluster requirements:

#### `slurm/train_qwen_vl.slurm`
- ✅ Added `#SBATCH --account=ingrai` (required for HAI cluster)
- ✅ Changed partition from `gpu` to `hai` (HAI cluster partition name)
- ✅ Configuration: 8 GPUs, 8 hours, 480GB RAM

#### `slurm/prepare_data.slurm`
- ✅ Added `#SBATCH --account=ingrai`
- ✅ No GPU partition needed (CPU-only job)
- ⚠️ **Note**: Consider using `./scripts/prepare_data.sh` instead to avoid SLURM overhead

### 2. Documentation

#### New Files
- **`HAI_CLUSTER_GUIDE.md`**: Comprehensive guide for HAI cluster
  - Storage locations and quotas
  - SLURM partitions and quotas
  - Setup and usage instructions
  - Troubleshooting tips
  - Performance recommendations

#### Updated Files
- **`QUICKSTART.md`**: Added HAI cluster reference and example commands

### 3. Key HAI Cluster Requirements

From the [official HAI cluster documentation](https://legacy.cs.stanford.edu/haic):

1. **Account Flag**: All SLURM jobs MUST include `--account=ingrai`
   ```bash
   sbatch --account=ingrai slurm/train_qwen_vl.slurm
   ```

2. **Partitions**:
   - `hai`: Main partition for production runs (8 hours is within default)
   - `hai-interactive`: For debugging (1 GPU max, 24hrs max)
   - `hai-lo`: Low priority, preemptible

3. **Quotas**:
   - Per-user: 8 GPUs, 8 running + 8 queued jobs
   - Per-account (ingrai team): 16 GPUs total, 16 running + 16 queued jobs
   - Your 8-GPU training job uses your full personal quota

4. **Storage**:
   - Home: `/hai/users/a/s/asanshay/` (50GB quota)
   - Scratch: `/hai/scratch/asanshay/` (5TB quota)
   - Dataset: `/hai/scratch/asanshay/websight-v2/data/`

5. **Important Rules**:
   - ⚠️ DO NOT run intensive processes on headnode `haic.stanford.edu`
   - Always use SLURM (`srun` or `sbatch`) for compute tasks
   - Coordinate with team members on quota usage

## Usage on HAI Cluster

### Quick Start

```bash
# 1. SSH to cluster (requires DUO 2FA)
ssh <sunetid>@haic.stanford.edu

# 2. Navigate to project
cd /hai/users/a/s/asanshay/websight-v2

# 3. Setup environment (one-time)
bash setup.sh
source venv/bin/activate

# 4. Prepare data
bash scripts/prepare_data.sh

# 5. Submit training job
sbatch --account=ingrai slurm/train_qwen_vl.slurm

# 6. Monitor
squeue -u $USER
tail -f logs/train_qwen3vl_*.out
```

### Testing Before Full Training

It's recommended to test on 1 GPU first:

```bash
# Get interactive GPU for debugging
srun --account=ingrai -p hai-interactive --gres=gpu:1 --time=1:00:00 --pty bash

# Activate environment
source venv/bin/activate

# Test training
python scripts/train.py --num-gpus 1 --no-deepspeed
```

### Monitoring Jobs

```bash
# Check your jobs
squeue -u $USER

# Check team quota usage
squeue -A ingrai

# Cancel a job
scancel <job_id>

# View detailed job info
scontrol show job <job_id>
```

## Performance Considerations

### H100 GPUs

The HAI cluster uses H100 GPUs with NVLink, configured for optimal performance:

1. **BF16 Training**: Enabled by default (better than FP16 for H100)
2. **Flash Attention 2**: Installed for faster attention computation
3. **DeepSpeed ZeRO-2**: Optimized for 8-GPU training
4. **NDR Infiniband**: Fast inter-GPU communication for distributed training

### Expected Performance

With the current configuration:
- Batch size: 2 per device × 4 grad accum × 8 GPUs = 64 effective batch size
- Throughput: ~10-15 samples/sec
- Training time: ~8 hours for 2.5 epochs on full dataset

### Tuning Tips

If you encounter memory issues:
```yaml
# In configs/qwen_vl_lora.yaml
per_device_train_batch_size: 1  # Reduce from 2
gradient_accumulation_steps: 8  # Increase from 4
```

If training is slower than expected:
- Check GPU utilization: `watch nvidia-smi`
- Verify data loading isn't bottlenecked
- Ensure Flash Attention is properly installed

## Common Issues

### "Permission denied" errors
```bash
# Verify dataset access
ls -lah /hai/scratch/asanshay/websight-v2/data/

# If missing, download it
bash scripts/download_dataset.sh
```

### "Unable to allocate resources"
```bash
# Check quota usage
squeue -u $USER  # Your jobs
squeue -A ingrai  # Team jobs

# Wait for jobs to complete or contact teammates
```

### "QOSMaxSubmitJobPerUserLimit"
- You've hit the 8 running + 8 queued limit
- Wait for jobs to complete before submitting more

### Job killed unexpectedly
```bash
# Check error logs
tail -100 logs/train_qwen3vl_<job_id>.err

# Common causes:
# - Out of memory: Reduce batch size
# - NCCL timeout: Check network, may be transient
# - CUDA OOM: Enable more aggressive checkpointing
```

## File Organization on HAI Cluster

```
/hai/users/a/s/asanshay/websight-v2/  (50GB quota)
├── venv/                          # Virtual environment
├── LLaMA-Factory/                # Training framework
├── configs/                      # Training configs
├── scripts/                      # Helper scripts
├── data/                         # Processed data (train/val)
├── logs/                         # SLURM job logs
├── saves/                        # Model checkpoints
└── .cache/                       # HuggingFace cache

/hai/scratch/asanshay/websight-v2/ (5TB quota)
└── data/                         # Raw dataset
    ├── prompts.jsonl
    └── images/
```

## Next Steps

1. **Initial Setup**: Run `setup.sh` to create environment and install dependencies
2. **Data Download**: Run `bash scripts/download_dataset.sh` if dataset isn't available
3. **Data Preparation**: Run `bash scripts/prepare_data.sh` to transform data
4. **Test Run**: Use `hai-interactive` partition to test on 1 GPU
5. **Full Training**: Submit with `sbatch --account=ingrai slurm/train_qwen_vl.slurm`

## Support

- **Cluster Issues**: https://support.cs.stanford.edu
- **Training Issues**: See `TRAINING_README.md` and `QUICKSTART.md`
- **Dataset Issues**: See `DATASET_INFO.md` and `DOWNLOAD_GUIDE.md`

---

**Last Updated**: November 11, 2025  
**Cluster**: Stanford HAI Compute Cluster  
**Account**: ingrai  
**Primary User**: asanshay

