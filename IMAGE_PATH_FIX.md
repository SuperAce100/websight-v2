# Image Path Fix - Separate Images Directory

## Problem

When using a separate images directory (e.g., `/hai/scratch/asanshay/screenspot-pro/images`), the inference was failing because:

1. Records in `data.jsonl` have paths like: `"image_path": "images/000000.png"`
2. The `load_screenspot_pro()` function returns the images directory path: `/hai/scratch/asanshay/screenspot-pro/images`
3. The inference script was joining them: `/hai/scratch/asanshay/screenspot-pro/images/images/000000.png` ❌

## Solution

Updated `run_screenspot_benchmark.py` to strip the `images/` prefix when the path starts with `images/`:

```python
# Handle both relative paths (images/000000.png) and direct paths (000000.png)
if image_path.startswith("images/"):
    # Strip "images/" prefix since media_dir already points to images directory
    image_filename = image_path.replace("images/", "", 1)
    full_image_path = os.path.join(media_dir, image_filename)
else:
    full_image_path = os.path.join(media_dir, image_path)
```

Now it correctly constructs: `/hai/scratch/asanshay/screenspot-pro/images/000000.png` ✅

## How to Use

### 1. Prepare Dataset (First Time)

```bash
# On Slurm (automatic)
sbatch slurm/screenspot_benchmark.slurm

# Or locally
python scripts/prepare_screenspot_pro.py \
    --output-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images
```

This will:
- Download ScreenSpot-Pro dataset
- Save `data.jsonl` to `screenspot_pro/`
- Save images to `/hai/scratch/asanshay/screenspot-pro/images/`

### 2. Run Inference

The Slurm script automatically handles the paths:
```bash
sbatch slurm/screenspot_benchmark.slurm
```

Or manually:
```bash
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --data-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images
```

## Verification

### Check Dataset is Prepared
```bash
# Check data.jsonl exists
ls -lh screenspot_pro/data.jsonl

# Check images exist
ls /hai/scratch/asanshay/screenspot-pro/images/ | head -10

# Count images
find /hai/scratch/asanshay/screenspot-pro/images/ -type f | wc -l
# Should show ~1582 images
```

### Check Image Paths in Data
```bash
# View first record
head -n 1 screenspot_pro/data.jsonl | python3 -m json.tool | grep image_path
# Should show: "image_path": "images/000000.png"
```

## Troubleshooting

### All Inferences Failing (100% failure rate)

**Symptom:**
```
Total samples: 497
✓ Successful: 0 (0.0%)
✗ Failed: 497 (100.0%)
```

**Cause:** Images not found

**Solution:**
1. Check if images directory exists:
   ```bash
   ls /hai/scratch/asanshay/screenspot-pro/images/
   ```

2. If empty or missing, re-download:
   ```bash
   python scripts/prepare_screenspot_pro.py \
       --output-dir screenspot_pro \
       --images-dir /hai/scratch/asanshay/screenspot-pro/images
   ```

### Some Images Not Found

**Symptom:**
```
Warning: Image not found: /hai/scratch/.../images/000123.png
  Media dir: /hai/scratch/asanshay/screenspot-pro/images
  Image path from record: images/000123.png
```

**Cause:** Incomplete download or corrupted images

**Solution:**
```bash
# Check which images are missing
python3 -c "
import json
from pathlib import Path

with open('screenspot_pro/data.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        img = rec['image_path'].replace('images/', '')
        path = Path('/hai/scratch/asanshay/screenspot-pro/images') / img
        if not path.exists():
            print(f'Missing: {img}')
"

# Re-download dataset
python scripts/prepare_screenspot_pro.py \
    --output-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images
```

## Debug Output

The script now shows debug info for the first 3 failures:
```
Warning: Image not found: /hai/scratch/asanshay/screenspot-pro/images/000000.png
  Media dir: /hai/scratch/asanshay/screenspot-pro/images
  Image path from record: images/000000.png
```

This helps diagnose path construction issues.

## Expected Behavior

### Successful Run
```
Running inference on 1582 samples...
  Output: runs/screenspot_pro/predictions_39031.jsonl
  Max new tokens: 512

  [1/1582] Success: 1, Failed: 0 | Accuracy: 0/1 (0.00%) | Rate: 0.19 samples/s
  [50/1582] Success: 49, Failed: 1 | Accuracy: 8/49 (16.33%) | Rate: 0.21 samples/s
  ...

================================================================================
Inference complete!
================================================================================
  Total samples: 1582
  ✓ Successful: 1580 (99.9%)
  ✗ Failed: 2 (0.1%)

  🎯 Accuracy: 298/1580 (18.86%)
  ⏱  Time: 8130.5s (135.5 min)
  📊 Rate: 0.19 samples/s
  💾 Predictions: runs/screenspot_pro/predictions_39031.jsonl
```

## Summary

The fix ensures that:
1. ✅ Records can use relative paths (`images/000000.png`)
2. ✅ Images can be stored in separate directory
3. ✅ Path construction works correctly
4. ✅ Both default and custom storage work seamlessly

Just make sure images are actually downloaded before running inference!

