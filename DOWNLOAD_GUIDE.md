# Dataset Download Guide

This guide explains how to download and set up the WebSight-v2 dataset at `/hai/scratch/websight-v2/data`.

## Quick Start

### Option 1: SLURM Job (Recommended)

1. **Configure the download source** by editing `slurm/download_dataset.slurm`:

```bash
# Open the file
nano slurm/download_dataset.slurm

# Uncomment and configure ONE of these options:

# For URL download:
DOWNLOAD_URL="https://example.com/websight-v2-dataset.tar.gz"

# For HuggingFace download:
HF_REPO="username/websight-v2"
HF_FILE="dataset.tar.gz"

# For local copy:
LOCAL_PATH="/existing/path/to/dataset"
```

2. **Submit the download job**:

```bash
sbatch slurm/download_dataset.slurm
```

3. **Monitor progress**:

```bash
# Check job status
squeue -u $USER

# Watch the log file
tail -f logs/download_dataset_*.out
```

### Option 2: Direct Python Script

For immediate download without SLURM:

```bash
# From URL
python scripts/download_dataset.py \
    --url https://your-dataset-url.com/dataset.tar.gz \
    --dest /hai/scratch/websight-v2/data

# From HuggingFace Hub
python scripts/download_dataset.py \
    --hf-repo username/websight-v2 \
    --hf-file dataset.tar.gz \
    --dest /hai/scratch/websight-v2/data

# Copy from local directory
python scripts/download_dataset.py \
    --local-path /path/to/existing/dataset \
    --dest /hai/scratch/websight-v2/data
```

## Download Methods

### Method 1: HTTP/HTTPS URL

Best for: Direct download links, web-hosted datasets

```bash
python scripts/download_dataset.py \
    --url https://example.com/dataset.tar.gz
```

**Supported formats:**
- `.tar.gz` / `.tgz`
- `.zip`
- `.tar`

### Method 2: HuggingFace Hub

Best for: Datasets hosted on HuggingFace

```bash
python scripts/download_dataset.py \
    --hf-repo username/dataset-name \
    --hf-file data.tar.gz
```

**Requirements:**
- `huggingface_hub` package (install: `pip install huggingface_hub`)
- HF authentication token if dataset is private (set via `huggingface-cli login`)

### Method 3: Local Copy

Best for: Dataset already exists on the cluster

```bash
python scripts/download_dataset.py \
    --local-path /path/to/existing/dataset
```

**Note**: This copies the data, so ensure you have sufficient disk space.

## Script Options

### Required (choose one)
- `--url URL`: Direct download URL
- `--hf-repo REPO --hf-file FILE`: HuggingFace repository and file
- `--local-path PATH`: Local directory to copy from

### Optional
- `--dest PATH`: Destination directory (default: `/hai/scratch/websight-v2/data`)
- `--temp-dir PATH`: Temporary directory for downloads (default: `/tmp/websight-v2-download`)
- `--keep-archive`: Keep the downloaded archive after extraction
- `--skip-verify`: Skip dataset verification after extraction

## What the Script Does

1. **Downloads** the dataset from the specified source
2. **Extracts** the archive to the destination directory
3. **Verifies** the dataset structure:
   - Checks for `prompts.jsonl`
   - Checks for `images/` directory
   - Validates JSON format
   - Counts records and images
4. **Reports** statistics and any issues

## Expected Dataset Structure

After extraction, the directory should contain:

```
/hai/scratch/websight-v2/data/
├── prompts.jsonl          # Main dataset file (~79k records)
└── images/                # Image directory
    ├── 000000.png
    ├── 000001.png
    ├── 000002.png
    └── ...
```

## Verification

After download, verify the dataset:

```bash
# Check files exist
ls -lh /hai/scratch/websight-v2/data/prompts.jsonl
ls /hai/scratch/websight-v2/data/images/ | head -n 10

# Count records (should be ~79,413)
wc -l /hai/scratch/websight-v2/data/prompts.jsonl

# View first record
head -n 1 /hai/scratch/websight-v2/data/prompts.jsonl | python -m json.tool
```

Expected output:
```
-rw-r--r-- ... 50M ... /hai/scratch/websight-v2/data/prompts.jsonl
79413 /hai/scratch/websight-v2/data/prompts.jsonl
```

## Troubleshooting

### Download fails

**Problem**: URL download fails with connection error

**Solution**:
- Check the URL is accessible: `curl -I https://your-url`
- Ensure you have internet access from compute nodes
- Try downloading to a different temp directory with `--temp-dir`

### HuggingFace authentication error

**Problem**: "Access denied" for HuggingFace dataset

**Solution**:
```bash
# Login to HuggingFace
pip install huggingface_hub
huggingface-cli login

# Enter your token when prompted
```

### Permission denied

**Problem**: Cannot write to `/hai/scratch/websight-v2/data`

**Solution**:
- Check parent directory exists and is writable
- Try downloading to a custom location: `--dest ~/my-dataset`
- Contact cluster administrator for `/hai/scratch/` access

### Extraction fails

**Problem**: Archive extraction errors

**Solution**:
- Verify archive isn't corrupted: check file size matches expected
- Ensure sufficient disk space (dataset needs ~15-20GB)
- Try with `--keep-archive` to inspect the downloaded file
- Manually extract and use `--local-path` to copy

### Dataset verification fails

**Problem**: "Invalid dataset structure" error

**Solution**:
- Check if archive has a nested directory structure
- Manually inspect the extracted files
- Ensure `prompts.jsonl` and `images/` are at the root level
- Use `--skip-verify` to bypass checks (not recommended)

### Out of disk space

**Problem**: No space left on device

**Solution**:
- Check available space: `df -h /hai/scratch/`
- Use a different temp directory: `--temp-dir /scratch/user/tmp`
- Clean up old files in temp directories
- Contact administrator to increase quota

## Example: Complete Workflow

```bash
# 1. Configure SLURM script
cat > slurm/download_dataset.slurm <<'EOF'
# ... (see slurm/download_dataset.slurm for full template)
DOWNLOAD_URL="https://example.com/websight-v2.tar.gz"
EOF

# 2. Submit download job
sbatch slurm/download_dataset.slurm

# 3. Wait for completion
squeue -u $USER

# 4. Verify
ls -lh /hai/scratch/websight-v2/data/
wc -l /hai/scratch/websight-v2/data/prompts.jsonl

# 5. Proceed with training data preparation
sbatch slurm/prepare_data.slurm
```

## Storage Requirements

- **Download**: ~10-15GB (compressed archive)
- **Extracted**: ~15-20GB (dataset files)
- **Temp space**: ~25-35GB (during extraction)
- **After cleanup**: ~15-20GB (just the dataset)

Make sure you have sufficient space in:
- Destination directory (`/hai/scratch/websight-v2/data`)
- Temporary directory (default: `/tmp`)

## Getting Help

If you encounter issues:

1. Check the error message in SLURM output: `cat logs/download_dataset_*.err`
2. Verify your configuration in `slurm/download_dataset.slurm`
3. Test with the Python script directly for better error messages
4. Check `DATASET_INFO.md` for dataset structure details
5. Contact your cluster administrator for storage/permission issues

## After Download

Once the dataset is at `/hai/scratch/websight-v2/data`, proceed with:

1. **Data preparation**: `sbatch slurm/prepare_data.slurm`
2. **Training**: `sbatch slurm/train_qwen_vl.slurm`

See `QUICKSTART.md` for the complete workflow.

