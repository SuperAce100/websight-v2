# Dataset Information

## Location

The training dataset is stored on the cluster at:

```
/hai/scratch/websight-v2/data/
```

## Structure

```
/hai/scratch/websight-v2/data/
├── prompts.jsonl          # Main dataset file (~79k samples)
└── images/                # Image directory
    ├── 000000.png
    ├── 000001.png
    ├── 000002.png
    └── ...
```

## Dataset Details

- **Total samples**: ~79,413 image-instruction-location pairs
- **File**: `prompts.jsonl` (JSONL format, one record per line)
- **Images**: PNG files in `images/` subdirectory
- **Sources**: Multiple UI datasets (webui, mind2web, motif, etc.)
- **Platforms**: Web and mobile interfaces

## Record Format

Each line in `prompts.jsonl` contains:

```json
{
  "id": 1,
  "original": {
    "instruction": "StaticText, link, listitem",
    "bbox": [792.46875, 56.0, 847.46875, 120.0],
    "resolution": [1280, 720],
    "source": "webui",
    "platform": "web",
    "name": "product link",
    "description": "A plain text link with the word 'Product'",
    "type": "link",
    "OCR": "Product",
    "language": "English",
    "purpose": "to navigate to the products page",
    "expectation": "the user will be directed to a page containing product information",
    "image_path": "images/000000.png",
    "split": "test"
  },
  "prompt": "click on the product link"
}
```

## Key Fields

- **id**: Unique identifier
- **original.bbox**: Bounding box [x_min, y_min, x_max, y_max] in original resolution
- **original.resolution**: Original image dimensions [width, height]
- **original.image_path**: Relative path to image (e.g., "images/000000.png")
- **original.split**: Train/test split designation
- **prompt**: Natural language instruction for the action

## Downloading the Dataset

If the dataset is not yet available at `/hai/scratch/websight-v2/data`, you can download it using the provided script.

### Using the Download Script

```bash
# From URL
python scripts/download_dataset.py --url https://example.com/dataset.tar.gz

# From HuggingFace Hub
python scripts/download_dataset.py --hf-repo username/websight-v2 --hf-file dataset.tar.gz

# Copy from local directory
python scripts/download_dataset.py --local-path /path/to/existing/dataset

# Via SLURM (recommended for large downloads)
sbatch slurm/download_dataset.slurm
```

**Note**: Edit `slurm/download_dataset.slurm` to configure your download source (URL, HuggingFace, or local path).

**See `DOWNLOAD_GUIDE.md` for detailed download instructions and troubleshooting.**

## Access

The dataset is read-only and shared across jobs. All scripts default to this location:

```bash
# Data transformation (default)
python scripts/transform_for_training.py
# Reads from: /hai/scratch/websight-v2/data/prompts.jsonl
# Images from: /hai/scratch/websight-v2/data/images/

# Override if needed
python scripts/transform_for_training.py \
    --input /custom/path/prompts.jsonl \
    --base-image-path /custom/path
```

## Training Data Output

The transformation script creates processed files in your workspace:

```
your-workspace/
└── data/
    ├── wave_ui_train.jsonl    # Transformed training set
    └── wave_ui_val.jsonl      # Transformed validation set
```

These reference images using absolute paths to `/hai/scratch/websight-v2/data/images/`.

## Verification

To verify dataset accessibility:

```bash
# Check if dataset exists
ls -lh /hai/scratch/websight-v2/data/prompts.jsonl

# Count records
wc -l /hai/scratch/websight-v2/data/prompts.jsonl

# View first record
head -n 1 /hai/scratch/websight-v2/data/prompts.jsonl | python -m json.tool

# Check images directory
ls /hai/scratch/websight-v2/data/images/ | head -n 10
```

## Storage Requirements

- **Source dataset**: ~15-20GB (images + JSONL)
- **Transformed data**: ~200-300MB (JSONL only, references source images)
- **Training workspace**: ~1-2GB (checkpoints, logs)

## Notes

- Images are shared and read-only - no copying needed
- The transformation script creates references, not copies
- All image paths in training data use absolute paths
- Dataset is accessible from all compute nodes

## Questions?

If you encounter dataset access issues:
1. Verify the path exists: `ls /hai/scratch/websight-v2/data/`
2. Check file permissions: `ls -la /hai/scratch/websight-v2/data/prompts.jsonl`
3. Ensure you're on a compute node with access to `/hai/scratch/`
4. Contact your cluster administrator if access is denied

