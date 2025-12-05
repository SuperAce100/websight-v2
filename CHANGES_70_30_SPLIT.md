# Changes Summary: 70/30 Data Split Implementation

## Overview

Implemented a 70/30 data split between SFT and KTO training to ensure no data overlap between the two training phases.

## Changes Made

### 1. `scripts/agentnet_scripts/transform_agentnet_for_training.py`

**Added new function** (lines 442-466):
```python
def split_sft_kto(records, kto_ratio=0.3, seed=42):
    """Split dataset into SFT and KTO portions."""
```

**Added new command-line arguments**:
- `--kto-ratio` (default: 0.3) - Percentage of data to reserve for KTO
- `--save-kto-split` - Flag to save the reserved KTO data to a file

**Modified main() function**:
- Data is now split FIRST into 70% SFT and 30% KTO
- The 70% SFT portion is then split into train/val/test
- The 30% KTO portion is saved to `data/agentnet_kto_split.jsonl` if `--save-kto-split` is used

**Result**: 
- No overlap between SFT and KTO data
- SFT uses 70% of trajectories (split into 80% train, 10% val, 10% test)
- KTO uses the remaining 30% of trajectories

### 2. `scripts/data_prep/prepare_kto_data.sh`

**Updated to use the KTO split file**:
- Now reads from `data/agentnet_kto_split.jsonl` (the reserved 30%)
- No longer processes the entire dataset
- Added clear error message if the split file is missing
- Provides instructions on how to generate the split file

### 3. `docs/KTO_TRAINING_SETUP.md`

**Added documentation**:
- Explained the 70/30 split strategy
- Added Step 0 to show how to prepare SFT data with the split
- Updated Step 1 to clarify it uses the reserved 30%
- Updated troubleshooting section with new workflow

### 4. `docs/DATA_SPLIT_70_30.md` (NEW FILE)

**Comprehensive documentation including**:
- Data flow diagram showing the split process
- Step-by-step implementation guide
- Detailed explanation of parameters
- Example calculations for 18K trajectories
- Verification methods to check for data overlap
- Troubleshooting tips
- Best practices

### 5. `README.md` (NEW FILE)

**Project overview including**:
- Quick start guide
- Links to all documentation
- Project structure
- Key features explanation

## Usage

### Complete Workflow

1. **Prepare data with 70/30 split**:
```bash
python scripts/agentnet_scripts/transform_agentnet_for_training.py \
  --input /path/to/agentnet_all.jsonl \
  --output-dir data \
  --base-image-dir /path/to/images \
  --kto-ratio 0.3 \
  --save-kto-split
```

**Outputs**:
- `data/agentnet_train.jsonl` - 56% of total (70% × 80%)
- `data/agentnet_val.jsonl` - 7% of total (70% × 10%)
- `data/agentnet_test.jsonl` - 7% of total (70% × 10%)
- `data/agentnet_kto_split.jsonl` - 30% of total (raw format)

2. **Train with SFT**:
```bash
sbatch --account=ingrai slurm/train_ui_tars_agentnet.slurm
```

3. **Prepare KTO data**:
```bash
bash scripts/data_prep/prepare_kto_data.sh
```

**Outputs**:
- `data/agentnet_kto.jsonl` - Transformed KTO format from the reserved 30%

4. **Train with KTO**:
```bash
sbatch --account=ingrai slurm/train_agentnet_kto.slurm
```

## Key Benefits

1. **No Data Leakage**: SFT and KTO use completely separate data
2. **Reproducible**: Using seed ensures consistent splits
3. **Flexible**: Can adjust `--kto-ratio` for different split ratios
4. **Well-Documented**: Comprehensive docs explain the process
5. **Backward Compatible**: Old workflow still works if you don't use `--kto-ratio`

## Verification

Check that the split worked correctly:

```bash
# Count lines in each file
wc -l data/agentnet_train.jsonl      # Should be ~56% of original
wc -l data/agentnet_val.jsonl        # Should be ~7% of original
wc -l data/agentnet_test.jsonl       # Should be ~7% of original
wc -l data/agentnet_kto_split.jsonl  # Should be ~30% of original
```

For 18,000 trajectories, expect approximately:
- agentnet_train.jsonl: ~10,080 trajectories
- agentnet_val.jsonl: ~1,260 trajectories
- agentnet_test.jsonl: ~1,260 trajectories
- agentnet_kto_split.jsonl: ~5,400 trajectories

## Important Notes

1. **Re-run if needed**: If you've already prepared data with the old version, re-run the transformation with the new parameters to get the proper split.

2. **Always use --save-kto-split**: Include this flag when preparing data so the KTO split is available later.

3. **Same seed for reproducibility**: Use `--seed 42` (or any fixed value) to ensure consistent splits across runs.

4. **Check for overlap**: The new split ensures no trajectory appears in both SFT and KTO datasets.

## Testing

To test with a small dataset:

```bash
python scripts/agentnet_scripts/transform_agentnet_for_training.py \
  --input test_data.jsonl \
  --output-dir test_output \
  --kto-ratio 0.3 \
  --save-kto-split \
  --max-examples-per-trajectory 5
```

## References

- Main transformation script: `scripts/agentnet_scripts/transform_agentnet_for_training.py`
- KTO preparation script: `scripts/data_prep/prepare_kto_data.sh`
- Detailed documentation: `docs/DATA_SPLIT_70_30.md`
- KTO training guide: `docs/KTO_TRAINING_SETUP.md`

