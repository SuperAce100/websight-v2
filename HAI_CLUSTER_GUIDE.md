# HAI Compute Cluster Guide

This guide covers specific setup and usage for the Stanford HAI Compute Cluster.

## Cluster Overview

- **Access**: SSH to `haic.stanford.edu` with your SUNetID (requires DUO 2FA)
- **OS**: Ubuntu 22.04
- **GPUs**: 40x Nvidia H100 with NVLink
- **Scheduler**: SLURM
- **Network**: NDR Infiniband for multi-node training

## Storage

Your storage locations on the HAI cluster:

```bash
# Home directory (50GB quota)
/hai/users/a/s/asanshay/

# Scratch space (5TB quota) - for datasets and checkpoints
/hai/scratch/asanshay/
```

**Dataset location**: `/hai/scratch/asanshay/websight-v2/data`

## SLURM Configuration

### Account Requirement

⚠️ **IMPORTANT**: All SLURM jobs MUST include `--account=ingrai`

```bash
# Example
sbatch --account=ingrai slurm/train_qwen_vl.slurm
srun --account=ingrai --gres=gpu:1 --pty bash
```

### Partitions

1. **`hai`** (default for batch jobs)
   - For production training runs
   - No shell access
   - Max walltime: 3 days (default: 24hrs)
   - All compute nodes available

2. **`hai-interactive`** 
   - For debugging and prototyping
   - Shell access allowed
   - Max walltime: 24hrs (default: 8hrs)
   - Limited to 1 GPU per user
   - Only 1 node (haic-hgx-1)

3. **`hai-lo`** (low priority)
   - Free-for-all when excess capacity available
   - Jobs may be preempted
   - Max walltime: 14 days (default: 24hrs)
   - No quotas

### Resource Quotas

**Per-user quota**:
- 8 GPUs max
- 8 running jobs + 8 queued jobs

**Per-account quota (ingrai team)**:
- 16 GPUs total for all team members
- 16 running jobs + 16 queued jobs

**Interactive partition**:
- 1 running job with 1 GPU max
- Counts toward your team quota

## Setup Instructions

### 1. Initial Setup (One-time)

```bash
# SSH to cluster (from campus network or VPN)
ssh <sunetid>@haic.stanford.edu

# Navigate to your workspace
cd /hai/users/a/s/asanshay/
git clone <your-repo-url> websight-v2
cd websight-v2

# Run setup script
bash setup.sh
```

### 2. Download Dataset (if not already available)

```bash
# Option A: Run locally (if data is on cluster already)
bash scripts/download_dataset.sh

# Option B: Submit as SLURM job (if downloading from internet)
sbatch --account=ingrai slurm/download_dataset.slurm
```

### 3. Prepare Training Data

```bash
# Option A: Run locally on login node (quick, ~5min)
bash scripts/prepare_data.sh

# Option B: Submit as SLURM job (safer for large datasets)
sbatch --account=ingrai slurm/prepare_data.slurm
```

### 4. Test Training (Interactive Debug)

Before submitting the full 8-hour job, test on 1 GPU:

```bash
# Request interactive GPU node
srun --account=ingrai -p hai-interactive --gres=gpu:1 --time=1:00:00 --pty bash

# Activate environment
source venv/bin/activate

# Test training script
python scripts/train.py --num-gpus 1 --no-deepspeed
```

### 5. Submit Full Training Job

```bash
# Submit 8-GPU training job (8 hours)
sbatch --account=ingrai slurm/train_qwen_vl.slurm
```

## Monitoring Jobs

```bash
# Check your jobs
squeue -u $USER

# Check all ingrai team jobs
squeue -A ingrai

# View job details
scontrol show job <job_id>

# View job output (while running)
tail -f logs/train_qwen3vl_<job_id>.out

# Cancel a job
scancel <job_id>
```

## Important Cluster Rules

1. **DO NOT run intensive processes on the headnode (`haic.stanford.edu`)**
   - No training, no heavy preprocessing
   - Process will be killed automatically
   - Use SLURM (`srun` or `sbatch`) for all compute tasks

2. **Use interactive partition for testing only**
   - Limited to 1 GPU
   - Short time limits
   - For debugging, not production runs

3. **Be mindful of team quotas**
   - Your team shares 16 GPU quota
   - Check team usage: `squeue -A ingrai`
   - Coordinate with teammates on large jobs

4. **Data transfer**
   - No dedicated data transfer node yet
   - Keep concurrent downloads reasonable
   - Use `rsync` or `scp` for large transfers

## Troubleshooting

### "Unable to allocate resources"

Check your quotas:
```bash
# Personal quota
squeue -u $USER

# Team quota (all ingrai members)
squeue -A ingrai
```

If at quota, wait for jobs to finish or cancel some jobs.

### "QOSMaxSubmitJobPerUserLimit"

You've hit the submission limit (8 running + 8 queued). Wait for jobs to complete.

### "Permission denied" on dataset

Check that dataset exists and you have access:
```bash
ls -lah /hai/scratch/asanshay/websight-v2/data/
```

If missing, run the download script.

### Job killed unexpectedly

Check error logs:
```bash
tail -100 logs/train_qwen3vl_<job_id>.err
```

Common causes:
- Out of memory: Reduce batch size in `configs/qwen_vl_lora.yaml`
- NCCL timeout: Check network connectivity
- CUDA OOM: Enable more aggressive gradient checkpointing

## Performance Tips

### For H100 GPUs:

1. **Use BF16 instead of FP16** (already configured)
   ```yaml
   bf16: true
   fp16: false
   ```

2. **Enable Flash Attention 2** (already configured)
   - Installed via `setup.sh`
   - Significantly faster attention computation

3. **Use DeepSpeed ZeRO-2** (already configured)
   - Configured in `configs/deepspeed_zero2.json`
   - Optimal for 8xH100 setup

4. **Monitor GPU utilization**
   ```bash
   # During training, check GPU usage
   watch nvidia-smi
   ```

5. **Adjust batch size if needed**
   - Current: 2 per device × 4 grad accum = 8 effective
   - Can try: 4 per device × 2 grad accum = 8 effective
   - Monitor memory with `nvidia-smi`

## File Organization

```
/hai/scratch/asanshay/websight-v2/
├── data/                          # Raw dataset (5TB quota)
│   ├── prompts.jsonl             # Original data
│   ├── images/                   # Image files
│   └── ...
│
/hai/users/a/s/asanshay/websight-v2/
├── venv/                          # Python virtual environment
├── LLaMA-Factory/                # Training framework
├── configs/                      # Training configs
├── scripts/                      # Utility scripts
├── data/                         # Processed training data
│   ├── wave_ui_train.jsonl
│   └── wave_ui_val.jsonl
├── logs/                         # SLURM logs
├── saves/                        # Model checkpoints
│   └── qwen3-vl-8b/
│       └── lora/
│           └── sft/
└── .cache/                       # HuggingFace cache

```

## Support

For cluster technical issues: <https://support.cs.stanford.edu>

For training/code issues: Check the main docs
- `QUICKSTART.md` - Quick reference
- `TRAINING_README.md` - Detailed training guide
- `SETUP_SUMMARY.md` - Configuration overview

