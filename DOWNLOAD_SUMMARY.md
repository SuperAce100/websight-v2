# Dataset Download Feature - Implementation Summary

## Overview

Added comprehensive dataset download and setup functionality to automatically pull the WebSight-v2 dataset to `/hai/scratch/websight-v2/data`.

## New Files Created

### 1. `scripts/download_dataset.py` (14 KB, executable)

**Full-featured Python script for dataset download and extraction.**

**Features:**
- ✅ Download from direct URLs (HTTP/HTTPS)
- ✅ Download from HuggingFace Hub
- ✅ Copy from local directories
- ✅ Support for multiple archive formats (tar.gz, zip, tar)
- ✅ Progress bars for downloads
- ✅ Automatic extraction with progress tracking
- ✅ Dataset structure verification
- ✅ Record and image counting
- ✅ Comprehensive error handling

**Usage:**
```bash
# From URL
python scripts/download_dataset.py --url https://example.com/dataset.tar.gz

# From HuggingFace
python scripts/download_dataset.py \
    --hf-repo username/websight-v2 \
    --hf-file dataset.tar.gz

# Copy from local
python scripts/download_dataset.py --local-path /path/to/dataset

# Custom destination
python scripts/download_dataset.py \
    --url https://example.com/dataset.tar.gz \
    --dest /custom/path
```

**Options:**
- `--url URL`: Direct download URL
- `--hf-repo REPO`: HuggingFace repository
- `--hf-file FILE`: File in HuggingFace repo
- `--local-path PATH`: Local directory to copy from
- `--dest PATH`: Destination (default: `/hai/scratch/websight-v2/data`)
- `--temp-dir PATH`: Temporary directory for downloads
- `--keep-archive`: Keep archive after extraction
- `--skip-verify`: Skip verification step

### 2. `slurm/download_dataset.slurm` (5.1 KB, executable)

**SLURM batch job for automated dataset download on the cluster.**

**Features:**
- ✅ 4-hour time limit (sufficient for large downloads)
- ✅ Configurable download source (URL/HuggingFace/local)
- ✅ Automatic environment setup
- ✅ Dataset existence checks
- ✅ Permission validation
- ✅ Comprehensive logging
- ✅ Automatic cleanup
- ✅ Post-download verification

**Configuration:**
Edit the script and set ONE of these:
```bash
# Option 1: URL download
DOWNLOAD_URL="https://your-url.com/dataset.tar.gz"

# Option 2: HuggingFace download  
HF_REPO="username/websight-v2"
HF_FILE="dataset.tar.gz"

# Option 3: Local copy
LOCAL_PATH="/existing/path/to/dataset"
```

**Usage:**
```bash
# 1. Configure the download source (edit the file)
nano slurm/download_dataset.slurm

# 2. Submit the job
sbatch slurm/download_dataset.slurm

# 3. Monitor progress
squeue -u $USER
tail -f logs/download_dataset_*.out
```

### 3. `DOWNLOAD_GUIDE.md` (7.1 KB)

**Complete guide for downloading and setting up the dataset.**

**Contents:**
- Quick start instructions
- Detailed explanation of all three download methods
- Command-line options reference
- Expected dataset structure
- Verification steps
- Comprehensive troubleshooting guide
- Storage requirements
- Complete workflow examples

## Updated Files

### Documentation Updates (6 files)

1. **`QUICKSTART.md`**
   - Added Step 0: Dataset download
   - Updated file list to include download scripts
   - Added reference to DOWNLOAD_GUIDE.md

2. **`TRAINING_README.md`**
   - Added Step 0: Download the Dataset section
   - Detailed download instructions with all methods
   - Updated workflow steps

3. **`SETUP_SUMMARY.md`**
   - Added Step 0 to manual setup
   - Download instructions before dependencies

4. **`DATASET_INFO.md`**
   - Added "Downloading the Dataset" section
   - Usage examples for download script
   - Reference to DOWNLOAD_GUIDE.md

5. **`QUICKSTART.md`** (references)
   - Added DOWNLOAD_GUIDE.md to "Need More Details?" section

## Features Summary

### Multi-Source Support

**1. Direct URL Download**
- Any HTTP/HTTPS URL
- Progress tracking
- Resume capability (partial)
- Works with tar.gz, zip, tar

**2. HuggingFace Hub**
- Native HuggingFace integration
- Private dataset support (with authentication)
- Optimized for HF infrastructure

**3. Local Copy**
- Copy from existing cluster location
- Useful for sharing datasets
- Preserves permissions and metadata

### Robust Error Handling

- ✅ Pre-flight checks (destination exists, writable, etc.)
- ✅ Download verification (size, integrity)
- ✅ Extraction error handling
- ✅ Structure validation
- ✅ Automatic cleanup on failure
- ✅ Clear error messages
- ✅ Graceful interruption (Ctrl+C)

### Verification System

After download, the script automatically verifies:
- `prompts.jsonl` exists and is valid JSON
- `images/` directory exists
- Record count matches expectations
- Image files are present
- Data format is correct

### Progress Tracking

**Download Progress:**
```
|████████████████████████████░░░░░░░░░░| 68.3% (10.24GB / 15.00GB)
```

**Extraction Progress:**
```
Extracted: 45000/79413 files
```

**Verification:**
```
✓ Found 79413 records in prompts.jsonl
✓ Found 79413 images
✓ Dataset verification passed!
```

## Complete Workflow

### Option 1: SLURM Job (Recommended)

```bash
# 1. Configure download source
nano slurm/download_dataset.slurm
# Set DOWNLOAD_URL, HF_REPO, or LOCAL_PATH

# 2. Submit download job
sbatch slurm/download_dataset.slurm

# 3. Wait for completion
squeue -u $USER
tail -f logs/download_dataset_*.out

# 4. Verify
ls -lh /hai/scratch/websight-v2/data/
wc -l /hai/scratch/websight-v2/data/prompts.jsonl

# 5. Proceed with training
sbatch slurm/prepare_data.slurm
sbatch slurm/train_qwen_vl.slurm
```

### Option 2: Direct Python Script

```bash
# Download directly (no SLURM)
python scripts/download_dataset.py \
    --url https://your-dataset-url.com/dataset.tar.gz

# Then proceed with training
python3 scripts/transform_for_training.py
python3 scripts/train.py --num-gpus 8
```

## Storage Requirements

| Stage | Space Required | Location |
|-------|---------------|----------|
| Download | ~10-15GB | Temp directory |
| Extraction | ~25-35GB | Temp + destination |
| Final | ~15-20GB | Destination only |

## Configuration Examples

### Example 1: Public URL

```bash
# In slurm/download_dataset.slurm:
DOWNLOAD_URL="https://example.com/websight-v2-dataset.tar.gz"

# Or directly:
python scripts/download_dataset.py \
    --url https://example.com/websight-v2-dataset.tar.gz
```

### Example 2: HuggingFace Hub

```bash
# In slurm/download_dataset.slurm:
HF_REPO="your-username/websight-v2"
HF_FILE="data.tar.gz"

# Or directly:
python scripts/download_dataset.py \
    --hf-repo your-username/websight-v2 \
    --hf-file data.tar.gz
```

### Example 3: Local Copy

```bash
# In slurm/download_dataset.slurm:
LOCAL_PATH="/scratch/shared/datasets/websight-v2"

# Or directly:
python scripts/download_dataset.py \
    --local-path /scratch/shared/datasets/websight-v2
```

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| Download fails | Check URL, internet access, try different temp-dir |
| HF auth error | Run `huggingface-cli login` |
| Permission denied | Check `/hai/scratch/` access, try custom --dest |
| Extraction fails | Verify archive integrity, check disk space |
| Verification fails | Check dataset structure, use --skip-verify if needed |
| Out of space | Clean temp files, increase quota, use different location |

## Integration with Existing Workflow

The download functionality seamlessly integrates with the existing training pipeline:

```
0. Download Dataset  ← NEW!
   ↓
1. Setup Environment
   ↓
2. Transform Data
   ↓
3. Train Model
```

All existing scripts automatically use `/hai/scratch/websight-v2/data` as the source location.

## Benefits

1. **Automated**: One command to download and set up
2. **Flexible**: Multiple download sources supported
3. **Robust**: Comprehensive error handling and verification
4. **Efficient**: Uses cluster resources optimally with SLURM
5. **User-Friendly**: Clear progress indicators and messages
6. **Safe**: Pre-checks prevent overwrites and errors
7. **Documented**: Complete guide with troubleshooting

## Documentation Structure

```
DOWNLOAD_GUIDE.md         - Complete download documentation
DOWNLOAD_SUMMARY.md       - This file (implementation overview)
DATASET_INFO.md           - Dataset structure and access info
QUICKSTART.md             - Quick start with download step
TRAINING_README.md        - Full training guide with download
SETUP_SUMMARY.md          - Setup overview with download
```

## Next Steps

Users can now:
1. ✅ Download the dataset from any source
2. ✅ Verify it's set up correctly
3. ✅ Proceed with data transformation
4. ✅ Start training immediately

All without manual intervention or copying files!

## Support

For help with dataset download:
- See `DOWNLOAD_GUIDE.md` for detailed instructions
- Check `DATASET_INFO.md` for dataset structure
- Review SLURM logs: `logs/download_dataset_*.out`
- Contact cluster admin for storage/permission issues

