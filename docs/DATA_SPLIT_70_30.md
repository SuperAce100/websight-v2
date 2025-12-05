# AgentNet Data Split: 70% SFT / 30% KTO

## Overview

The AgentNet dataset is split into two non-overlapping portions:
- **70% for Supervised Fine-Tuning (SFT)**: Used for initial model training
- **30% for KTO (Kahneman-Tversky Optimization)**: Used for preference-based refinement

This split ensures that the model is not evaluated or refined on data it has already seen during SFT.

## Data Flow Diagram

```
agentnet_all.jsonl (100% of data)
        |
        v
[transform_agentnet_for_training.py with --kto-ratio 0.3]
        |
        +---> 70% SFT portion
        |     |
        |     +---> 80% train --> data/agentnet_train.jsonl
        |     +---> 10% val   --> data/agentnet_val.jsonl
        |     +---> 10% test  --> data/agentnet_test.jsonl
        |
        +---> 30% KTO portion --> data/agentnet_kto_split.jsonl
              |
              v
        [transform_agentnet_for_kto.py]
              |
              v
        data/agentnet_kto.jsonl
```

## Implementation

### Step 1: Prepare All Data with Split

Run the SFT transformation script with the `--save-kto-split` flag:

```bash
python scripts/agentnet_scripts/transform_agentnet_for_training.py \
  --input /hai/scratch/asanshay/websight-v2/agentnet/agentnet_all.jsonl \
  --output-dir data \
  --base-image-dir /hai/scratch/asanshay/websight-v2/agentnet/images \
  --kto-ratio 0.3 \
  --save-kto-split \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

**What this does:**
1. Loads all AgentNet trajectories
2. Randomly splits into 70% SFT and 30% KTO (using seed 42 for reproducibility)
3. Saves the 30% KTO split to `data/agentnet_kto_split.jsonl` (raw format)
4. Further splits the 70% SFT portion into:
   - 80% train → `data/agentnet_train.jsonl` (transformed format)
   - 10% val → `data/agentnet_val.jsonl` (transformed format)
   - 10% test → `data/agentnet_test.jsonl` (transformed format)

**Output files:**
- `data/agentnet_train.jsonl` - ~56% of total data (70% × 80%)
- `data/agentnet_val.jsonl` - ~7% of total data (70% × 10%)
- `data/agentnet_test.jsonl` - ~7% of total data (70% × 10%)
- `data/agentnet_kto_split.jsonl` - 30% of total data (reserved for KTO)

### Step 2: Transform KTO Data

After completing SFT training, transform the reserved KTO split:

```bash
bash scripts/data_prep/prepare_kto_data.sh
```

Or manually:

```bash
python scripts/agentnet_scripts/transform_agentnet_for_kto.py \
  --input data/agentnet_kto_split.jsonl \
  --output data/agentnet_kto.jsonl \
  --base-image-dir /hai/scratch/asanshay/websight-v2/agentnet/images
```

**What this does:**
1. Reads the reserved 30% of trajectories from `data/agentnet_kto_split.jsonl`
2. Transforms them into KTO format with preference labels (kto_tag)
3. Saves to `data/agentnet_kto.jsonl`

### Step 3: Train with SFT

```bash
sbatch --account=ingrai slurm/train_ui_tars_agentnet.slurm
```

Uses: `data/agentnet_train.jsonl` and `data/agentnet_val.jsonl`

### Step 4: Train with KTO

```bash
sbatch --account=ingrai slurm/train_agentnet_kto.slurm
```

Uses: `data/agentnet_kto.jsonl` (from the reserved 30%)

## Key Parameters

### `--kto-ratio` (default: 0.3)
Controls what percentage of data is reserved for KTO. Common values:
- `0.3` (default) - 70% SFT, 30% KTO
- `0.2` - 80% SFT, 20% KTO
- `0.4` - 60% SFT, 40% KTO

### `--val-ratio` and `--test-ratio` (default: 0.1 each)
Controls the split *within the SFT portion*:
- `--val-ratio 0.1` means 10% of SFT data goes to validation
- `--test-ratio 0.1` means 10% of SFT data goes to test
- Remaining ~80% goes to training

### `--seed` (default: 42)
Random seed for reproducibility. Using the same seed ensures consistent splits across runs.

### `--save-kto-split`
Flag to save the raw KTO split file. Required if you plan to run KTO training later.

## Example with 18K Trajectories

For a dataset with 18,000 trajectories:

| Split | Trajectories | Percentage | Purpose |
|-------|-------------|-----------|---------|
| **SFT Train** | ~12,600 | 70% | Primary training data |
| **SFT Val** | ~1,260 | 7% | Validation during SFT |
| **SFT Test** | ~1,260 | 7% | Final SFT evaluation |
| **KTO** | ~5,400 | 30% | Preference optimization |
| **Total** | 18,000 | 100% | - |

If each trajectory has ~4 steps on average:
- SFT training examples: ~50,400 (12,600 × 4)
- KTO examples: ~5,400 (one per trajectory)

## Verification

### Check file sizes

```bash
# Count trajectories in each file
wc -l data/agentnet_train.jsonl
wc -l data/agentnet_val.jsonl
wc -l data/agentnet_test.jsonl
wc -l data/agentnet_kto_split.jsonl

# The KTO split should be ~30% of the total
# train + val + test should be ~70% of the total
```

### Check for overlap

To verify there's no overlap between SFT and KTO data, you can check task IDs:

```bash
# Extract task IDs from SFT files
cat data/agentnet_train.jsonl data/agentnet_val.jsonl data/agentnet_test.jsonl | \
  python -c "import sys, json; [print(json.loads(line).get('task_id', '')) for line in sys.stdin if line.strip()]" | \
  sort | uniq > sft_task_ids.txt

# Extract task IDs from KTO split
python -c "import sys, json; [print(json.loads(line).get('task_id', '')) for line in open('data/agentnet_kto_split.jsonl') if line.strip()]" | \
  sort | uniq > kto_task_ids.txt

# Check for overlap (should be empty)
comm -12 sft_task_ids.txt kto_task_ids.txt
```

## Troubleshooting

### I already ran the old version without --save-kto-split

Re-run the transformation with the new parameters:

```bash
python scripts/agentnet_scripts/transform_agentnet_for_training.py \
  --input /path/to/agentnet_all.jsonl \
  --output-dir data \
  --kto-ratio 0.3 \
  --save-kto-split
```

This will overwrite the existing files with the new 70/30 split.

### I want a different split ratio

Adjust the `--kto-ratio` parameter:

```bash
# For 80% SFT / 20% KTO
python scripts/agentnet_scripts/transform_agentnet_for_training.py \
  --input /path/to/agentnet_all.jsonl \
  --output-dir data \
  --kto-ratio 0.2 \
  --save-kto-split
```

### The files were split before this change

If you've already trained on the old split, you should:
1. Re-run the transformation with the new code
2. Re-train SFT from scratch (to ensure proper 70/30 split)
3. Then run KTO training

## Best Practices

1. **Always use `--save-kto-split`** when preparing data for eventual KTO training
2. **Use the same `--seed`** value if you need to regenerate splits
3. **Keep the raw KTO split file** (`agentnet_kto_split.jsonl`) for reproducibility
4. **Document your split ratio** in your experiment notes
5. **Complete SFT training** before preparing KTO data (though the files can be prepared in advance)

## References

- SFT transformation script: `scripts/agentnet_scripts/transform_agentnet_for_training.py`
- KTO transformation script: `scripts/agentnet_scripts/transform_agentnet_for_kto.py`
- KTO preparation script: `scripts/data_prep/prepare_kto_data.sh`
- KTO training guide: `docs/KTO_TRAINING_SETUP.md`

