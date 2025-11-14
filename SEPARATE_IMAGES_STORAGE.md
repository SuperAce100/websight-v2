# Separate Images Storage Configuration

The ScreenSpot-Pro benchmark now supports storing images in a separate directory from `data.jsonl`, which is useful for cluster environments where you want to store large image files on scratch storage.

## Configuration

### Default Behavior
By default, images are stored in `output-dir/images/`:
```bash
python scripts/prepare_screenspot_pro.py --output-dir screenspot_pro
# Creates:
#   screenspot_pro/data.jsonl
#   screenspot_pro/images/
```

### Separate Images Directory
To store images in a different location (e.g., scratch storage):
```bash
python scripts/prepare_screenspot_pro.py \
    --output-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images

# Creates:
#   screenspot_pro/data.jsonl
#   /hai/scratch/asanshay/screenspot-pro/images/
```

## Slurm Configuration

The Slurm job script automatically uses `/hai/scratch/asanshay/screenspot-pro/images` as the default images directory:

```bash
# Default configuration (images on scratch)
sbatch slurm/screenspot_benchmark.slurm
```

### Custom Images Directory
Override the default with an environment variable:
```bash
sbatch --export=IMAGES_DIR="/custom/path/to/images" slurm/screenspot_benchmark.slurm
```

### Custom Data Directory
Change where `data.jsonl` is stored:
```bash
sbatch --export=DATA_DIR="/custom/data/dir",IMAGES_DIR="/custom/images/dir" \
    slurm/screenspot_benchmark.slurm
```

## Running Inference

When running inference, specify both directories if images are stored separately:

```bash
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --data-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images
```

## Verification

The scripts automatically verify that both `data.jsonl` and images exist:

### Dataset Preparation
```
✓ Dataset preparation complete!
================================================================================
  Records: 1582
  Images: 1582
  Data file: screenspot_pro/data.jsonl
  Media directory: /hai/scratch/asanshay/screenspot-pro/images
```

### Slurm Job
```
Step 1: Prepare ScreenSpot-Pro Dataset
========================================

✓ Dataset already exists at screenspot_pro/data.jsonl
  Samples: 1582
  Images: 1582
```

If image count doesn't match sample count, the script will warn you and optionally re-download.

## Directory Structure

### With Separate Storage
```
workspace/
├── screenspot_pro/
│   ├── data.jsonl              # Metadata and annotations
│   └── raw/                    # Cached raw dataset
│
/hai/scratch/asanshay/screenspot-pro/
└── images/                     # Large image files
    ├── 000000.png
    ├── 000001.png
    └── ...
```

### With Default Storage
```
workspace/
└── screenspot_pro/
    ├── data.jsonl              # Metadata and annotations
    ├── images/                 # Image files
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── raw/                    # Cached raw dataset
```

## Benefits

1. **Scratch Storage**: Store large image files on high-speed scratch storage
2. **Separation of Concerns**: Keep metadata (data.jsonl) in workspace, images on scratch
3. **Flexibility**: Easy to move images without re-downloading
4. **Shared Access**: Multiple users can share the same images directory

## Environment Variables (Slurm)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `screenspot_pro` | Directory for data.jsonl |
| `IMAGES_DIR` | `/hai/scratch/asanshay/screenspot-pro/images` | Directory for images |

## Complete Example

### 1. Prepare Dataset with Separate Storage
```bash
python scripts/prepare_screenspot_pro.py \
    --output-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images
```

### 2. Run Benchmark Locally
```bash
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --data-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images
```

### 3. Or Submit to Slurm (Automatic)
```bash
sbatch slurm/screenspot_benchmark.slurm
```

The Slurm job automatically uses the correct paths!

## Troubleshooting

### Images Not Found
If you get "Image not found" errors:

1. Check that `IMAGES_DIR` is set correctly
2. Verify images exist:
   ```bash
   ls /hai/scratch/asanshay/screenspot-pro/images/ | head
   ```
3. Check image count:
   ```bash
   find /hai/scratch/asanshay/screenspot-pro/images/ -type f | wc -l
   ```

### Re-download Images
If images are missing or corrupted:
```bash
# Remove old images
rm -rf /hai/scratch/asanshay/screenspot-pro/images/

# Re-download
python scripts/prepare_screenspot_pro.py \
    --output-dir screenspot_pro \
    --images-dir /hai/scratch/asanshay/screenspot-pro/images
```

## Notes

- The scripts automatically create parent directories if they don't exist
- Image count is verified against sample count (with tolerance of ±10)
- Both local and Slurm execution support separate storage
- The `data.jsonl` file references images as `images/000000.png`, etc. (relative paths)

