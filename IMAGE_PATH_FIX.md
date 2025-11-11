# Image Path Configuration Fix

## Issue

The dataset images are stored separately at `/hai/scratch/asanshay/websight-v2/data/images/`, not in the workspace directory.

## Solution

### 1. Transform Script (`scripts/transform_for_training.py`)

**Changed**: Store only relative paths in the transformed JSONL files
```python
# Before (absolute path):
"images": [f"{base_image_path}/{image_path}"]
# Example: "/hai/scratch/asanshay/websight-v2/data/images/screenshot_123.png"

# After (relative path):
"images": [image_path]  
# Example: "images/screenshot_123.png"
```

### 2. Training Script (`slurm/train_qwen_vl.slurm`)

**Added**: `--image_folder` parameter pointing to the dataset base directory
```bash
--image_folder /hai/scratch/asanshay/websight-v2/data \
```

### How It Works

LLaMA-Factory combines the paths:
```
--image_folder: /hai/scratch/asanshay/websight-v2/data
+ image path:   images/screenshot_123.png
= Full path:    /hai/scratch/asanshay/websight-v2/data/images/screenshot_123.png ✓
```

## What You Need to Do

Since you've already prepared the data with the old format (absolute paths), you have two options:

### Option A: Re-prepare the data (Recommended)

```bash
# On the cluster, re-run data preparation with the updated script
bash scripts/prepare_data.sh
```

This will regenerate `wave_ui_train.jsonl` and `wave_ui_val.jsonl` with relative paths.

### Option B: Use the existing data

The training script should work with absolute paths too (LLaMA-Factory detects absolute paths and uses them as-is). However, Option A is cleaner.

## Verification

After re-preparing data, check the image paths:

```bash
# Should show relative path like: "images/screenshot_123.png"
head -1 data/wave_ui_train.jsonl | python3 -c "import json, sys; data=json.load(sys.stdin); print('Image path:', data['images'][0])"
```

Expected output:
```
Image path: images/screenshot_123.png
```

NOT:
```
Image path: /hai/scratch/asanshay/websight-v2/data/images/screenshot_123.png
```

## Files Modified

1. **`scripts/transform_for_training.py`**
   - Line 104: Changed to store relative paths only
   - Removed `base_image_path` from the images field

2. **`slurm/train_qwen_vl.slurm`**
   - Line 185: Added `--image_folder /hai/scratch/asanshay/websight-v2/data`

## Next Steps

```bash
# 1. Re-prepare data with fixed script
bash scripts/prepare_data.sh

# 2. Verify the paths are relative
head -1 data/wave_ui_train.jsonl | python3 -m json.tool | grep -A 1 "images"

# 3. Submit training job
sbatch --account=ingrai slurm/train_qwen_vl.slurm
```

## Troubleshooting

### If you see "Image not found" errors during training:

Check that the images actually exist:
```bash
# Extract an image path from your data
IMAGE_PATH=$(head -1 data/wave_ui_train.jsonl | python3 -c "import json, sys; print(json.load(sys.stdin)['images'][0])")

# Check if file exists
ls -lh "/hai/scratch/asanshay/websight-v2/data/${IMAGE_PATH}"
```

### If paths are still absolute after re-preparing:

Make sure you're using the updated `transform_for_training.py` script:
```bash
# Check the script
grep -A 2 '"images":' scripts/transform_for_training.py
# Should show: "images": [image_path],
```

