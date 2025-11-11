# Dataset Path Migration Summary

## Overview

All scripts and documentation have been updated to use the centralized dataset location at `/hai/scratch/websight-v2/data`.

## What Changed

### ✅ Updated Default Paths

**Before:**
- Scripts expected data in local `wave-ui/` directory
- Required manual path specification
- Mixed relative/absolute paths

**After:**
- All scripts default to `/hai/scratch/websight-v2/data`
- Consistent absolute paths throughout
- SLURM scripts include path validation

## Modified Files

### 1. Scripts

#### `scripts/transform_for_training.py`
```python
# Changed defaults:
--input: "wave-ui/prompts.jsonl" → "/hai/scratch/websight-v2/data/prompts.jsonl"
--base-image-path: "wave-ui" → "/hai/scratch/websight-v2/data"
```

### 2. SLURM Scripts

#### `slurm/prepare_data.slurm`
- Added dataset verification check
- Updated workspace directory to use `${SLURM_SUBMIT_DIR}`
- Set `DATA_DIR` environment variable
- Validates dataset exists before processing

```bash
DATA_DIR="/hai/scratch/websight-v2/data"

# Verify dataset exists
if [ ! -f "${DATA_DIR}/prompts.jsonl" ]; then
    echo "✗ Error: Dataset not found at ${DATA_DIR}/prompts.jsonl"
    exit 1
fi
```

#### `slurm/train_qwen_vl.slurm`
- Updated workspace directory handling
- Added `DATA_DIR` variable for consistency

### 3. Documentation

#### `TRAINING_README.md`
- Updated data preparation commands
- Added note about dataset location
- Updated directory structure diagram

#### `SETUP_SUMMARY.md`
- Updated dataset format examples
- Changed image paths to absolute
- Added dataset location note

#### `QUICKSTART.md`
- Updated dataset references
- Added dataset location note
- Updated alternative testing section

#### `CHANGES.md`
- Updated example paths

#### `setup_training.sh`
- Added dataset location in output
- Added verification step

### 4. New Documentation

#### `DATASET_INFO.md` (NEW)
Complete documentation covering:
- Dataset location and structure
- Record format details
- Access instructions
- Verification commands
- Storage requirements
- Troubleshooting tips

## Benefits

1. **Consistency**: All scripts use the same centralized location
2. **Validation**: SLURM scripts verify dataset exists before running
3. **Flexibility**: Workspace location is dynamic (`${SLURM_SUBMIT_DIR}`)
4. **Documentation**: Clear reference in `DATASET_INFO.md`
5. **No Copying**: Images referenced directly, no duplication

## Usage

### Default Behavior (Recommended)

Simply run scripts without arguments - they'll use the cluster dataset:

```bash
# Data preparation
python scripts/transform_for_training.py
# or
sbatch slurm/prepare_data.slurm

# Training
sbatch slurm/train_qwen_vl.slurm
```

### Custom Dataset Location (If Needed)

Override defaults with command-line arguments:

```bash
python scripts/transform_for_training.py \
    --input /custom/path/prompts.jsonl \
    --base-image-path /custom/path
```

## Verification

Check dataset accessibility:

```bash
# Verify dataset exists
ls -lh /hai/scratch/websight-v2/data/prompts.jsonl

# Count records (should be ~79,413)
wc -l /hai/scratch/websight-v2/data/prompts.jsonl

# Check images directory
ls /hai/scratch/websight-v2/data/images/ | wc -l
```

Expected output:
```
-rw-r--r-- 1 user group 50M Nov 10 16:00 /hai/scratch/websight-v2/data/prompts.jsonl
79413 /hai/scratch/websight-v2/data/prompts.jsonl
79413
```

## Migration for Existing Users

If you previously prepared data with local paths:

1. **No action needed** - old transformed data still works
2. **To update**: Simply re-run data preparation:
   ```bash
   python scripts/transform_for_training.py
   ```
3. New files will reference cluster dataset location
4. Training will use the updated paths

## Dataset Structure

```
/hai/scratch/websight-v2/data/
├── prompts.jsonl          # Source dataset (~79k records)
└── images/                # Image files
    ├── 000000.png
    ├── 000001.png
    └── ... (79,413 images)
```

Your workspace:
```
your-workspace/
└── data/
    ├── wave_ui_train.jsonl    # Processed training data
    └── wave_ui_val.jsonl      # Processed validation data
```

Training data references images using absolute paths:
```json
{
  "images": ["/hai/scratch/websight-v2/data/images/000000.png"]
}
```

## Troubleshooting

### Dataset Not Found

```bash
# Check if path exists
ls -ld /hai/scratch/websight-v2/data/

# Check permissions
ls -la /hai/scratch/websight-v2/data/prompts.jsonl

# Verify you're on a compute node with access
srun --pty bash
ls /hai/scratch/websight-v2/data/
```

### Permission Denied

- Ensure you have read access to `/hai/scratch/`
- Contact cluster admin if needed
- Check if you're in the correct user group

### Wrong Record Count

```bash
# Should be exactly 79413
wc -l /hai/scratch/websight-v2/data/prompts.jsonl

# If different, dataset may be corrupted or wrong version
# Contact dataset maintainer
```

## Additional Resources

- `DATASET_INFO.md` - Detailed dataset documentation
- `TRAINING_README.md` - Full training guide
- `QUICKSTART.md` - Quick start instructions

## Questions?

For dataset-related issues:
1. Check `DATASET_INFO.md`
2. Verify path and permissions
3. Contact cluster administrator
4. Check SLURM job logs for specific errors

