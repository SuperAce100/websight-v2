# Plot Training Curves

Generate training visualizations from checkpoint directories or TensorBoard logs.

## Quick Start

### From Checkpoint Directory

```bash
python scripts/plot_training_curves.py \
    --checkpoint_dir ckpts/checkpoint-200 \
    --output_dir training_plots
```

### From Multiple Checkpoints (Comparison)

```bash
python scripts/plot_training_curves.py \
    --checkpoint_dir ckpts/checkpoint-200 \
    --checkpoint_dir ckpts/checkpoint-400 \
    --checkpoint_dir ckpts/checkpoint-600 \
    --output_dir training_plots
```

### From TensorBoard Logs

```bash
python scripts/plot_training_curves.py \
    --tensorboard_dir logs/qwen3-vl-8b \
    --output_dir training_plots
```

### Combined (Checkpoint + TensorBoard)

```bash
python scripts/plot_training_curves.py \
    --checkpoint_dir ckpts/checkpoint-200 \
    --tensorboard_dir logs/qwen3-vl-8b \
    --output_dir training_plots
```

## Output Files

The script generates three plots:

1. **loss_curves.png** - Training and evaluation loss curves
2. **learning_rate_curve.png** - Learning rate schedule over training
3. **training_metrics.png** - Combined view of all metrics (4 subplots)

## Options

- `--checkpoint_dir`: Path to checkpoint directory (can specify multiple times)
- `--tensorboard_dir`: Path to TensorBoard log directory
- `--output_dir`: Output directory for plots (default: `training_plots`)
- `--format`: Output format - `png`, `pdf`, or `svg` (default: `png`)
- `--dpi`: DPI for output images (default: `300`)

## Examples

### Basic Usage

```bash
# Single checkpoint
python scripts/plot_training_curves.py --checkpoint_dir ckpts/checkpoint-200
```

### Compare Multiple Runs

```bash
python scripts/plot_training_curves.py \
    --checkpoint_dir saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200 \
    --checkpoint_dir saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-400 \
    --output_dir comparison_plots
```

### High Resolution PDF

```bash
python scripts/plot_training_curves.py \
    --checkpoint_dir ckpts/checkpoint-200 \
    --format pdf \
    --dpi 600 \
    --output_dir training_plots
```

## What Gets Plotted

- **Training Loss**: Loss over training steps
- **Evaluation Loss**: Validation loss (if available)
- **Learning Rate**: LR schedule over training
- **Epoch Progress**: Training progress in epochs

## Training Summary

The script also prints a summary:
- Initial/Final/Min loss values
- Learning rate range
- Total steps and epochs
- Best checkpoint information

## Requirements

```bash
pip install matplotlib numpy
# Optional: for TensorBoard support
pip install tensorboard
```

## Notes

- The script reads `trainer_state.json` from checkpoint directories
- For TensorBoard logs, it looks for `events.out.tfevents.*` files
- Multiple checkpoints are plotted on the same axes for easy comparison
- All plots are saved as high-resolution PNG files (300 DPI by default)

