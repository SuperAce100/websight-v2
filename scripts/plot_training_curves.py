#!/usr/bin/env python3
"""
Generate training curves from trainer_state.json or TensorBoard logs.

This script can:
1. Read trainer_state.json from checkpoint directories
2. Read TensorBoard event files
3. Generate loss curves, learning rate curves, and other training metrics
4. Support multiple runs for comparison

Usage:
    # From checkpoint directory
    python scripts/plot_training_curves.py --checkpoint_dir ckpts/checkpoint-200

    # From TensorBoard logs
    python scripts/plot_training_curves.py --tensorboard_dir logs/qwen3-vl-8b

    # Multiple checkpoints for comparison
    python scripts/plot_training_curves.py \
        --checkpoint_dir ckpts/checkpoint-200 \
        --checkpoint_dir ckpts/checkpoint-400 \
        --output_dir training_plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")


def load_trainer_state(checkpoint_dir: Path) -> Optional[Dict]:
    """Load trainer_state.json from checkpoint directory."""
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    
    if not trainer_state_path.exists():
        print(f"⚠ Warning: trainer_state.json not found in {checkpoint_dir}")
        return None
    
    try:
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"❌ Error loading trainer_state.json: {e}")
        return None


def extract_metrics_from_trainer_state(state: Dict) -> Dict[str, List]:
    """Extract training metrics from trainer_state.json."""
    metrics = {
        'train_loss': [],
        'learning_rate': [],
        'epoch': [],
        'step': [],
        'grad_norm': []
    }
    
    # Extract log history
    log_history = state.get('log_history', [])
    
    for entry in log_history:
        step = entry.get('step', None)
        epoch = entry.get('epoch', None)
        
        if 'loss' in entry and 'eval_loss' not in entry:
            metrics['train_loss'].append((step, entry['loss']))
            if epoch is not None:
                metrics['epoch'].append((step, epoch))
        
        if 'grad_norm' in entry:
            metrics['grad_norm'].append((step, entry['grad_norm']))
        
        if 'learning_rate' in entry:
            metrics['learning_rate'].append((step, entry['learning_rate']))
    
    return metrics


def load_tensorboard_logs(log_dir: Path) -> Optional[Dict[str, List]]:
    """Load metrics from TensorBoard event files."""
    if not TENSORBOARD_AVAILABLE:
        print("⚠ Warning: tensorboard not available. Skipping TensorBoard logs.")
        return None
    
    if not log_dir.exists():
        print(f"⚠ Warning: TensorBoard log directory not found: {log_dir}")
        return None
    
    try:
        # Find event files
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            print(f"⚠ Warning: No TensorBoard event files found in {log_dir}")
            return None
        
        # Use the most recent event file
        event_file = sorted(event_files, key=lambda x: x.stat().st_mtime)[-1]
        
        # Load events
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        
        metrics = {
            'train_loss': [],
            'learning_rate': [],
            'step': []
        }
        
        # Extract scalar metrics
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            for event in scalar_events:
                step = int(event.step)
                value = float(event.value)
                
                if 'loss' in tag.lower() and 'train' in tag.lower() and 'eval' not in tag.lower():
                    metrics['train_loss'].append((step, value))
                elif 'learning_rate' in tag.lower() or 'lr' in tag.lower():
                    metrics['learning_rate'].append((step, value))
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error loading TensorBoard logs: {e}")
        return None


def plot_loss_curves(
    metrics_list: List[Tuple[str, Dict[str, List]]],
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 6)
):
    """Plot loss curves from multiple runs."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training loss only
    for name, metrics in metrics_list:
        if metrics['train_loss']:
            steps, losses = zip(*sorted(metrics['train_loss']))
            ax.plot(steps, losses, label=name, marker='o', markersize=2, linewidth=1.5)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved loss curves to: {output_path}")
    plt.close()


def plot_learning_rate_curve(
    metrics_list: List[Tuple[str, Dict[str, List]]],
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 6)
):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, metrics in metrics_list:
        if metrics['learning_rate']:
            steps, lrs = zip(*sorted(metrics['learning_rate']))
            ax.plot(steps, lrs, label=name, marker='o', markersize=2, linewidth=1.5)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for learning rate
    
    plt.tight_layout()
    output_path = output_dir / 'learning_rate_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning rate curve to: {output_path}")
    plt.close()


def plot_combined_metrics(
    metrics_list: List[Tuple[str, Dict[str, List]]],
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 8)
):
    """Plot all metrics in a comprehensive figure."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Training loss
    ax1 = axes[0, 0]
    for name, metrics in metrics_list:
        if metrics['train_loss']:
            steps, losses = zip(*sorted(metrics['train_loss']))
            ax1.plot(steps, losses, label=name, marker='o', markersize=1, linewidth=1)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient norm (if available)
    ax2 = axes[0, 1]
    has_grad_norm = False
    for name, metrics in metrics_list:
        if 'grad_norm' in metrics and metrics['grad_norm']:
            steps, norms = zip(*sorted(metrics['grad_norm']))
            ax2.plot(steps, norms, label=name, marker='s', markersize=1, linewidth=1)
            has_grad_norm = True
    if has_grad_norm:
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'Gradient norm data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Gradient Norm')
    
    # Learning rate
    ax3 = axes[1, 0]
    for name, metrics in metrics_list:
        if metrics['learning_rate']:
            steps, lrs = zip(*sorted(metrics['learning_rate']))
            ax3.plot(steps, lrs, label=name, marker='o', markersize=1, linewidth=1)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Epoch progress
    ax4 = axes[1, 1]
    for name, metrics in metrics_list:
        if metrics['epoch']:
            steps, epochs = zip(*sorted(metrics['epoch']))
            ax4.plot(steps, epochs, label=name, marker='o', markersize=1, linewidth=1)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Epoch')
    ax4.set_title('Training Progress')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined metrics to: {output_path}")
    plt.close()


def print_training_summary(metrics: Dict[str, List], name: str):
    """Print a summary of training metrics."""
    print(f"\n{'='*60}")
    print(f"Training Summary: {name}")
    print(f"{'='*60}")
    
    if metrics['train_loss']:
        steps, losses = zip(*sorted(metrics['train_loss']))
        print(f"Training Loss:")
        print(f"  Initial: {losses[0]:.4f}")
        print(f"  Final:   {losses[-1]:.4f}")
        print(f"  Min:     {min(losses):.4f} (at step {steps[losses.index(min(losses))]})")
        print(f"  Steps:   {len(losses)}")
    
    if metrics['learning_rate']:
        steps, lrs = zip(*sorted(metrics['learning_rate']))
        print(f"\nLearning Rate:")
        print(f"  Initial: {lrs[0]:.2e}")
        print(f"  Final:   {lrs[-1]:.2e}")
        print(f"  Max:     {max(lrs):.2e}")
    
    if metrics['epoch']:
        steps, epochs = zip(*sorted(metrics['epoch']))
        print(f"\nTraining Progress:")
        print(f"  Total Steps: {max(steps)}")
        print(f"  Total Epochs: {max(epochs):.2f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate training curves from checkpoint or TensorBoard logs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        action='append',
        help="Path to checkpoint directory (can specify multiple times)"
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_plots",
        help="Output directory for plots (default: training_plots)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output images (default: 300)"
    )
    
    args = parser.parse_args()
    
    if not args.checkpoint_dir and not args.tensorboard_dir:
        parser.error("Must specify at least one of --checkpoint_dir or --tensorboard_dir")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_list = []
    
    # Load from checkpoint directories
    if args.checkpoint_dir:
        for checkpoint_path in args.checkpoint_dir:
            checkpoint_dir = Path(checkpoint_path)
            name = checkpoint_dir.name
            
            print(f"Loading metrics from checkpoint: {checkpoint_dir}")
            state = load_trainer_state(checkpoint_dir)
            
            if state:
                metrics = extract_metrics_from_trainer_state(state)
                metrics_list.append((name, metrics))
                print_training_summary(metrics, name)
            else:
                print(f"⚠ Skipping {checkpoint_dir}")
    
    # Load from TensorBoard logs
    if args.tensorboard_dir:
        log_dir = Path(args.tensorboard_dir)
        print(f"Loading metrics from TensorBoard: {log_dir}")
        metrics = load_tensorboard_logs(log_dir)
        
        if metrics:
            name = log_dir.name or "tensorboard"
            metrics_list.append((name, metrics))
            print_training_summary(metrics, name)
    
    if not metrics_list:
        print("❌ Error: No metrics found!")
        return 1
    
    # Generate plots
    print(f"\nGenerating plots in {output_dir}...")
    
    # Loss curves
    plot_loss_curves(metrics_list, output_dir)
    
    # Learning rate curve
    plot_learning_rate_curve(metrics_list, output_dir)
    
    # Combined metrics
    plot_combined_metrics(metrics_list, output_dir)
    
    print(f"\n{'='*60}")
    print("✓ All plots generated successfully!")
    print(f"{'='*60}")
    print(f"\nPlots saved to: {output_dir}")
    print(f"  - loss_curves.png")
    print(f"  - learning_rate_curve.png")
    print(f"  - training_metrics.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

