# ScreenSpot-Pro Benchmark

This directory contains scripts to reproduce the ScreenSpot-Pro benchmark using the `Asanshay/websight-v2-grounded` model (or any compatible vision-language model).

## Overview

The benchmark workflow consists of three main steps:

1. **Dataset Preparation**: Download and transform the ScreenSpot-Pro dataset from HuggingFace
2. **Inference**: Run model inference on the dataset to generate predictions
3. **Evaluation**: Compare predictions against ground truth bounding boxes to compute accuracy

## Quick Start

### Local Execution

```bash
# 1. Prepare the dataset
python scripts/prepare_screenspot_pro.py --output-dir screenspot_pro

# 2. Run inference with the default model
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --data-dir screenspot_pro \
    --output runs/screenspot_pro/predictions.jsonl

# 3. Evaluate predictions
python scripts/evaluate_screenspot.py \
    --predictions runs/screenspot_pro/predictions.jsonl \
    --ground-truth screenspot_pro/data.jsonl
```

### Slurm Cluster Execution

```bash
# Run the complete benchmark pipeline on Slurm
sbatch slurm/screenspot_benchmark.slurm

# Or with custom model
sbatch --export=MODEL_PATH="your-org/your-model" slurm/screenspot_benchmark.slurm

# Or with LoRA adapter
sbatch --export=MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct",ADAPTER_PATH="ckpts/checkpoint-200" \
    slurm/screenspot_benchmark.slurm

# Test on subset (first 100 samples)
sbatch --export=LIMIT=100 slurm/screenspot_benchmark.slurm
```

## Scripts

### 1. Dataset Preparation

#### `scripts/screenspot_pro_utils.py`

Shared utility module for downloading and loading the ScreenSpot-Pro dataset.

**Key Functions:**
- `download_screenspot_pro()`: Downloads dataset from HuggingFace and transforms to ShareGPT format
- `load_screenspot_pro()`: Loads existing dataset from disk

**Features:**
- Automatic retry with exponential backoff for rate limiting
- Incremental writing (progress preserved if interrupted)
- Preserves all metadata (bbox, application, platform, ui_type, etc.)
- Creates numbered image files for easy reference

#### `scripts/prepare_screenspot_pro.py`

CLI script for dataset preparation.

**Usage:**
```bash
# Download and prepare dataset
python scripts/prepare_screenspot_pro.py --output-dir screenspot_pro

# Load existing dataset (skip download)
python scripts/prepare_screenspot_pro.py --skip-download --output-dir screenspot_pro

# Custom retry settings
python scripts/prepare_screenspot_pro.py --max-retries 10 --retry-delay 120
```

**Arguments:**
- `--output-dir`: Output directory for processed dataset (default: `screenspot_pro`)
- `--skip-download`: Skip download if dataset already exists
- `--max-retries`: Maximum retry attempts for rate limiting (default: 5)
- `--retry-delay`: Initial delay between retries in seconds (default: 60)

**Output:**
- `screenspot_pro/data.jsonl`: Transformed dataset in ShareGPT format
- `screenspot_pro/images/`: Directory containing all images
- `screenspot_pro/raw/`: Raw downloaded dataset (cached)

### 2. Inference

#### `scripts/run_screenspot_benchmark.py`

Run inference on ScreenSpot-Pro dataset using HuggingFace models.

**Usage:**
```bash
# Default model (Asanshay/websight-v2-grounded)
python scripts/run_screenspot_benchmark.py \
    --data-dir screenspot_pro

# Custom model
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path your-org/your-model \
    --data-dir screenspot_pro

# With LoRA adapter
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --adapter-path ckpts/checkpoint-200 \
    --data-dir screenspot_pro

# Test on subset
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --data-dir screenspot_pro \
    --limit 100
```

**Arguments:**
- `--model-name-or-path`: HuggingFace model ID or local path (default: `Asanshay/websight-v2-grounded`)
- `--adapter-path`: Optional path to LoRA adapter
- `--data-dir`: Directory containing prepared dataset (default: `screenspot_pro`)
- `--output`: Output file for predictions (default: `runs/screenspot_pro/<timestamp>.jsonl`)
- `--device`: Device to run inference on (default: `cuda`)
- `--max-new-tokens`: Maximum tokens to generate (default: 512)
- `--limit`: Limit number of samples for testing
- `--progress-interval`: Print progress every N samples (default: 50)

**Output:**
- Predictions JSONL file with one prediction per line
- Each prediction contains: sample_id, instruction, image_path, output, bbox, metadata

### 3. Evaluation

#### `scripts/evaluate_screenspot.py`

Evaluate predictions against ground truth bounding boxes.

**Usage:**
```bash
# Basic evaluation
python scripts/evaluate_screenspot.py \
    --predictions runs/screenspot_pro/predictions.jsonl \
    --ground-truth screenspot_pro/data.jsonl

# With verbose output and JSON export
python scripts/evaluate_screenspot.py \
    --predictions runs/screenspot_pro/predictions.jsonl \
    --ground-truth screenspot_pro/data.jsonl \
    --verbose \
    --output-json results.json
```

**Arguments:**
- `--predictions`: Path to predictions JSONL file (required)
- `--ground-truth`: Path to ground truth JSONL file (required)
- `--verbose`: Print detailed results for each sample
- `--output-json`: Save results to JSON file

**Metrics Computed:**
- **Overall Accuracy**: Percentage of predictions where click point falls within ground truth bbox
- **Parse Rate**: Percentage of outputs successfully parsed as coordinates
- **Distance Statistics**: Average and median distance from prediction to bbox
- **Breakdown by Category**: Accuracy by application, platform, UI type, and group

**Output Format:**
```
ScreenSpot-Pro Evaluation Results
================================================================================

Overall Metrics:
--------------------------------------------------------------------------------
  Total samples:        1582
  Parsed successfully:  1580 (99.9%)
  Failed to parse:      2 (0.1%)
  Missing ground truth: 0

  ✓ Correct:            298 / 1580 (18.86%)
  ✗ Incorrect:          1282 / 1580 (81.14%)

  Average distance:     125.3 pixels
  Median distance:      89.0 pixels

Breakdown by Application:
--------------------------------------------------------------------------------
  origin                          45 /  250 (18.0%)
  blender                         38 /  200 (19.0%)
  ...
```

### 4. Slurm Workflow

#### `slurm/screenspot_benchmark.slurm`

End-to-end Slurm job that chains dataset preparation, inference, and evaluation.

**Usage:**
```bash
# Default: Use Asanshay/websight-v2-grounded model
sbatch slurm/screenspot_benchmark.slurm

# Custom model
sbatch --export=MODEL_PATH="your-org/your-model" slurm/screenspot_benchmark.slurm

# With LoRA adapter
sbatch --export=MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct",ADAPTER_PATH="ckpts/checkpoint-200" \
    slurm/screenspot_benchmark.slurm

# Test on subset
sbatch --export=LIMIT=100 slurm/screenspot_benchmark.slurm

# Custom data directory
sbatch --export=DATA_DIR="/scratch/screenspot_pro" slurm/screenspot_benchmark.slurm
```

**Environment Variables:**
- `MODEL_PATH`: Model to use (default: `Asanshay/websight-v2-grounded`)
- `ADAPTER_PATH`: Optional LoRA adapter path
- `DATA_DIR`: Dataset directory (default: `screenspot_pro`)
- `LIMIT`: Limit number of samples for testing
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: 512)

**Resource Requirements:**
- 1 GPU
- 64GB RAM
- 8 CPUs
- 8 hours time limit

**Output Files:**
- `logs/screenspot_benchmark_<job_id>.out`: Job output log
- `logs/screenspot_benchmark_<job_id>.err`: Job error log
- `runs/screenspot_pro/predictions_<job_id>.jsonl`: Predictions
- `runs/screenspot_pro/results_<job_id>.json`: Evaluation results

## Dataset Format

### Input Format (data.jsonl)

Each line is a JSON object with the following structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in using electronic devices..."
    },
    {
      "role": "user",
      "content": "<image>\nQuery: plot a candlestick chart\nOutput only the coordinate..."
    }
  ],
  "image_path": "images/000000.png",
  "images": ["images/000000.png"],
  "instruction": "plot a candlestick chart",
  "sample_id": 0,
  "bbox": [300, 2077, 324, 2100],
  "gt_bbox": [300, 2077, 324, 2100],
  "id": "origin_windows_0",
  "application": "origin",
  "platform": "windows",
  "img_size": [3840, 2160],
  "ui_type": "icon",
  "group": "Scientific"
}
```

### Prediction Format (predictions.jsonl)

Each line is a JSON object with the following structure:

```json
{
  "sample_id": 0,
  "instruction": "plot a candlestick chart",
  "image_path": "images/000000.png",
  "output": "pyautogui.click(312, 2088)",
  "bbox": [300, 2077, 324, 2100],
  "application": "origin",
  "platform": "windows",
  "ui_type": "icon",
  "group": "Scientific"
}
```

### Evaluation Results Format (results.json)

```json
{
  "total": 1582,
  "parsed": 1580,
  "unparsed": 2,
  "correct": 298,
  "incorrect": 1282,
  "missing_gt": 0,
  "accuracy": 18.86,
  "avg_distance": 125.3,
  "median_distance": 89.0,
  "by_application": {
    "origin": {"total": 250, "correct": 45},
    "blender": {"total": 200, "correct": 38}
  },
  "by_platform": {...},
  "by_ui_type": {...},
  "by_group": {...}
}
```

## Model Requirements

Models must be compatible with:
- `transformers.AutoModelForVision2Seq`
- `transformers.AutoProcessor`
- Support for conversation format with system and user messages
- Support for `<image>` token in user messages

Tested with:
- `Asanshay/websight-v2-grounded` (default)
- `Qwen/Qwen3-VL-8B-Instruct` (with or without LoRA adapters)

## Troubleshooting

### Rate Limiting

If you encounter rate limiting errors when downloading the dataset:

1. Login to HuggingFace CLI:
   ```bash
   huggingface-cli login
   ```

2. Increase retry settings:
   ```bash
   python scripts/prepare_screenspot_pro.py --max-retries 10 --retry-delay 120
   ```

### Out of Memory

If you run out of GPU memory during inference:

1. Reduce batch size (currently 1, cannot be reduced further)
2. Use a smaller model
3. Request more GPU memory in Slurm:
   ```bash
   sbatch --mem=128G slurm/screenspot_benchmark.slurm
   ```

### Parse Failures

If predictions fail to parse:

1. Check model output format in predictions file
2. Verify model is generating `pyautogui.click(x, y)` format
3. Update parser in `evaluate_screenspot.py` if needed

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{screenspot-pro,
  title={ScreenSpot-Pro: Evaluating GUI Grounding in Professional Environments},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This benchmark implementation is provided under the same license as the websight-v2 project.
The ScreenSpot-Pro dataset is subject to its own license terms from HuggingFace.

