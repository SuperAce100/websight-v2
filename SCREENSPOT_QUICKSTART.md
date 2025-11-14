# ScreenSpot-Pro Benchmark - Quick Reference

## One-Line Commands

### Local Execution (Full Pipeline)
```bash
# Prepare dataset, run inference, and evaluate
python scripts/prepare_screenspot_pro.py && \
python scripts/run_screenspot_benchmark.py --model-name-or-path Asanshay/websight-v2-grounded && \
python scripts/evaluate_screenspot.py --predictions runs/screenspot_pro/predictions_*.jsonl --ground-truth screenspot_pro/data.jsonl
```

### Slurm Execution (Full Pipeline)
```bash
# Submit job with default model
sbatch slurm/screenspot_benchmark.slurm

# Monitor job
squeue -u $USER
tail -f logs/screenspot_benchmark_<job_id>.out
```

## Common Use Cases

### 1. Test Your Model
```bash
# Replace with your model path
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path your-org/your-model \
    --data-dir screenspot_pro \
    --output my_predictions.jsonl

python scripts/evaluate_screenspot.py \
    --predictions my_predictions.jsonl \
    --ground-truth screenspot_pro/data.jsonl
```

### 2. Test on Small Subset (Quick)
```bash
# Test on first 10 samples
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --limit 10

# Or via Slurm
sbatch --export=LIMIT=10 slurm/screenspot_benchmark.slurm
```

### 3. Use LoRA Adapter
```bash
# Local
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --adapter-path ckpts/checkpoint-200

# Slurm
sbatch --export=MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct",ADAPTER_PATH="ckpts/checkpoint-200" \
    slurm/screenspot_benchmark.slurm
```

### 4. Re-evaluate Existing Predictions
```bash
python scripts/evaluate_screenspot.py \
    --predictions runs/screenspot_pro/predictions_12345.jsonl \
    --ground-truth screenspot_pro/data.jsonl \
    --output-json results.json
```

### 5. Skip Dataset Download (Use Cached)
```bash
python scripts/prepare_screenspot_pro.py --skip-download
```

## File Locations

```
websight-v2/
├── scripts/
│   ├── screenspot_pro_utils.py          # Shared utilities
│   ├── prepare_screenspot_pro.py        # Dataset preparation
│   ├── run_screenspot_benchmark.py      # Inference runner
│   └── evaluate_screenspot.py           # Evaluation script
├── slurm/
│   └── screenspot_benchmark.slurm       # End-to-end Slurm job
├── screenspot_pro/                      # Dataset (created by prepare script)
│   ├── data.jsonl                       # Transformed dataset
│   ├── images/                          # Image files
│   └── raw/                             # Raw downloaded data (cached)
├── runs/screenspot_pro/                 # Results (created by benchmark)
│   ├── predictions_<timestamp>.jsonl    # Model predictions
│   └── results_<timestamp>.json         # Evaluation metrics
└── logs/                                # Slurm logs
    ├── screenspot_benchmark_<job_id>.out
    └── screenspot_benchmark_<job_id>.err
```

## Expected Results

### Asanshay/websight-v2-grounded Baseline
- **Dataset**: ~1,582 samples
- **Expected Accuracy**: 15-25% (ScreenSpot-Pro is challenging)
- **Runtime**: ~2-4 hours on single GPU
- **GPU Memory**: ~16-24GB

### Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| Rate limiting during download | `huggingface-cli login` then retry |
| Out of GPU memory | Use smaller model or request more memory |
| Parse failures | Check model output format in predictions file |
| Dataset not found | Run `prepare_screenspot_pro.py` first |
| Slurm job fails | Check logs in `logs/screenspot_benchmark_<job_id>.err` |

## Environment Variables (Slurm)

```bash
# Model configuration
export MODEL_PATH="Asanshay/websight-v2-grounded"
export ADAPTER_PATH=""                    # Optional LoRA adapter

# Dataset configuration
export DATA_DIR="screenspot_pro"

# Inference configuration
export LIMIT=""                           # Optional sample limit
export MAX_NEW_TOKENS="512"

# Submit with custom config
sbatch --export=ALL,MODEL_PATH="your-model" slurm/screenspot_benchmark.slurm
```

## Performance Tips

1. **Cache Dataset**: Run `prepare_screenspot_pro.py` once, reuse for multiple runs
2. **Use Slurm**: Automatic retry, logging, and resource management
3. **Test First**: Use `--limit 10` to verify setup before full run
4. **Monitor Progress**: Check logs in real-time with `tail -f`
5. **Save Results**: Use `--output-json` to save metrics for comparison

## Next Steps

After running the benchmark:

1. **Analyze Results**: Check breakdown by application, platform, UI type
2. **Compare Models**: Run multiple models and compare accuracy
3. **Error Analysis**: Use `--verbose` to see which samples failed
4. **Improve Model**: Fine-tune on failure cases
5. **Share Results**: Document accuracy and runtime for reproducibility

For detailed documentation, see [SCREENSPOT_BENCHMARK.md](SCREENSPOT_BENCHMARK.md)

