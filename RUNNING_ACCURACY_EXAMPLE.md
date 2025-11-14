# Running Accuracy Display - Example Output

The benchmark runner now displays **running accuracy** as it processes samples, giving you real-time feedback on model performance.

## Example Output

```
================================================================================
ScreenSpot-Pro Benchmark
================================================================================
Model: Asanshay/websight-v2-grounded
Data directory: screenspot_pro
Output: runs/screenspot_pro/predictions_20241114_150000.jsonl
Device: cuda
================================================================================

Loading existing dataset from screenspot_pro/data.jsonl...
Loaded 1582 records
Loading model from Asanshay/websight-v2-grounded...
  ✓ Processor loaded
  ✓ Model loaded and set to eval mode

Running inference on 1582 samples...
  Output: runs/screenspot_pro/predictions_20241114_150000.jsonl
  Max new tokens: 512

  [1/1582] Success: 1, Failed: 0 | Accuracy: 0/1 (0.00%) | Rate: 0.19 samples/s
  [50/1582] Success: 49, Failed: 1 | Accuracy: 8/49 (16.33%) | Rate: 0.21 samples/s
  [100/1582] Success: 98, Failed: 2 | Accuracy: 17/98 (17.35%) | Rate: 0.20 samples/s
  [150/1582] Success: 148, Failed: 2 | Accuracy: 26/148 (17.57%) | Rate: 0.20 samples/s
  [200/1582] Success: 197, Failed: 3 | Accuracy: 35/197 (17.77%) | Rate: 0.19 samples/s
  [250/1582] Success: 247, Failed: 3 | Accuracy: 45/247 (18.22%) | Rate: 0.19 samples/s
  [300/1582] Success: 296, Failed: 4 | Accuracy: 54/296 (18.24%) | Rate: 0.19 samples/s
  [350/1582] Success: 346, Failed: 4 | Accuracy: 64/346 (18.50%) | Rate: 0.19 samples/s
  [400/1582] Success: 395, Failed: 5 | Accuracy: 73/395 (18.48%) | Rate: 0.19 samples/s
  [450/1582] Success: 445, Failed: 5 | Accuracy: 83/445 (18.65%) | Rate: 0.19 samples/s
  [500/1582] Success: 494, Failed: 6 | Accuracy: 92/494 (18.62%) | Rate: 0.19 samples/s
  [550/1582] Success: 544, Failed: 6 | Accuracy: 102/544 (18.75%) | Rate: 0.19 samples/s
  [600/1582] Success: 593, Failed: 7 | Accuracy: 111/593 (18.72%) | Rate: 0.19 samples/s
  [650/1582] Success: 643, Failed: 7 | Accuracy: 121/643 (18.82%) | Rate: 0.19 samples/s
  [700/1582] Success: 692, Failed: 8 | Accuracy: 130/692 (18.79%) | Rate: 0.19 samples/s
  [750/1582] Success: 742, Failed: 8 | Accuracy: 140/742 (18.87%) | Rate: 0.19 samples/s
  [800/1582] Success: 791, Failed: 9 | Accuracy: 149/791 (18.84%) | Rate: 0.19 samples/s
  [850/1582] Success: 841, Failed: 9 | Accuracy: 159/841 (18.91%) | Rate: 0.19 samples/s
  [900/1582] Success: 890, Failed: 10 | Accuracy: 168/890 (18.88%) | Rate: 0.19 samples/s
  [950/1582] Success: 940, Failed: 10 | Accuracy: 178/940 (18.94%) | Rate: 0.19 samples/s
  [1000/1582] Success: 989, Failed: 11 | Accuracy: 187/989 (18.91%) | Rate: 0.19 samples/s
  [1050/1582] Success: 1039, Failed: 11 | Accuracy: 197/1039 (18.96%) | Rate: 0.19 samples/s
  [1100/1582] Success: 1088, Failed: 12 | Accuracy: 206/1088 (18.93%) | Rate: 0.19 samples/s
  [1150/1582] Success: 1138, Failed: 12 | Accuracy: 216/1138 (18.98%) | Rate: 0.19 samples/s
  [1200/1582] Success: 1187, Failed: 13 | Accuracy: 225/1187 (18.96%) | Rate: 0.19 samples/s
  [1250/1582] Success: 1237, Failed: 13 | Accuracy: 235/1237 (19.00%) | Rate: 0.19 samples/s
  [1300/1582] Success: 1286, Failed: 14 | Accuracy: 244/1286 (18.98%) | Rate: 0.19 samples/s
  [1350/1582] Success: 1336, Failed: 14 | Accuracy: 254/1336 (19.01%) | Rate: 0.19 samples/s
  [1400/1582] Success: 1385, Failed: 15 | Accuracy: 263/1385 (18.99%) | Rate: 0.19 samples/s
  [1450/1582] Success: 1435, Failed: 15 | Accuracy: 273/1435 (19.03%) | Rate: 0.19 samples/s
  [1500/1582] Success: 1484, Failed: 16 | Accuracy: 282/1484 (19.00%) | Rate: 0.19 samples/s
  [1550/1582] Success: 1534, Failed: 16 | Accuracy: 292/1534 (19.03%) | Rate: 0.19 samples/s

================================================================================
Inference complete!
================================================================================
  Total samples: 1582
  ✓ Successful: 1580 (99.9%)
  ✗ Failed: 2 (0.1%)

  🎯 Accuracy: 298/1580 (18.86%)
  ⏱  Time: 8130.5s (135.5 min)
  📊 Rate: 0.19 samples/s
  💾 Predictions: runs/screenspot_pro/predictions_20241114_150000.jsonl

Next steps:
  Evaluate predictions:
    python scripts/evaluate_screenspot.py \
      --predictions runs/screenspot_pro/predictions_20241114_150000.jsonl \
      --ground-truth screenspot_pro/data.jsonl
```

## What's Displayed

### During Inference (every 50 samples by default)
- **Progress**: Current sample / Total samples
- **Success/Failed**: Number of successful inferences vs failures
- **Accuracy**: Correct predictions / Evaluated samples (percentage)
  - Only counts samples where coordinates were parsed and bbox was available
  - Updates in real-time as inference progresses
- **Rate**: Samples processed per second

### Final Summary
- Total samples processed
- Success/failure breakdown
- **Final Accuracy**: Overall accuracy across all samples
- Total time and processing rate
- Output file location

## Configuration

### Change Progress Interval
```bash
# Show progress every 10 samples instead of 50
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --progress-interval 10
```

### Test on Small Subset
```bash
# Test on first 100 samples to see accuracy quickly
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path Asanshay/websight-v2-grounded \
    --limit 100 \
    --progress-interval 10
```

## Benefits

1. **Real-time Feedback**: See how your model is performing without waiting for completion
2. **Early Detection**: Spot issues (low accuracy, high failure rate) early
3. **Progress Tracking**: Know exactly how long the benchmark will take
4. **Immediate Results**: Get accuracy estimate before running full evaluation

## Notes

- Running accuracy is computed on-the-fly during inference
- Only samples with valid coordinates and ground truth bbox are counted
- The final accuracy should match the detailed evaluation from `evaluate_screenspot.py`
- Progress updates don't slow down inference (minimal overhead)

