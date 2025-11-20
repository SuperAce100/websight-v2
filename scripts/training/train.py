#!/usr/bin/env python3
"""
Main training script for Qwen3-VL fine-tuning using LLaMA-Factory.

This script launches the training process with the specified configuration.
It's designed to work with SLURM and distributed training on multiple GPUs.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def setup_environment():
    """Set up environment variables for training."""
    # Enable TF32 for better performance on H100
    os.environ["CUDA_TF32_ENABLED"] = "1"
    
    # Set up HuggingFace cache if needed
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    
    # Disable tokenizers parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Environment setup complete")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} found")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("✗ PyTorch not found!")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} found")
    except ImportError:
        print("✗ Transformers not found!")
        return False
    
    try:
        import deepspeed
        print(f"✓ DeepSpeed {deepspeed.__version__} found")
    except ImportError:
        print("⚠ DeepSpeed not found (optional but recommended)")
    
    # Check for LLaMA-Factory
    llama_factory_path = Path("LLaMA-Factory")
    if llama_factory_path.exists():
        print(f"✓ LLaMA-Factory found at {llama_factory_path}")
    else:
        print(f"✗ LLaMA-Factory not found at {llama_factory_path}")
        print("  Please clone it with: git clone https://github.com/hiyouga/LLaMA-Factory.git")
        return False
    
    return True


def run_training(
    config_path: str,
    dataset_info: str,
    num_gpus: int = 8,
    use_deepspeed: bool = True,
    resume_from_checkpoint: str = None
):
    """
    Launch the training process.
    
    Args:
        config_path: Path to training configuration YAML
        dataset_info: Path to dataset_info.json
        num_gpus: Number of GPUs to use
        use_deepspeed: Whether to use DeepSpeed
        resume_from_checkpoint: Optional path to checkpoint to resume from
    """
    # Get absolute paths
    config_path = Path(config_path).resolve()
    dataset_info = Path(dataset_info).resolve()
    workspace = Path.cwd()
    
    # LLaMA-Factory training command
    llamafactory_path = workspace / "LLaMA-Factory"
    
    if not llamafactory_path.exists():
        print(f"Error: LLaMA-Factory not found at {llamafactory_path}")
        sys.exit(1)
    
    # Build command
    if use_deepspeed and num_gpus > 1:
        # Use DeepSpeed launcher for multi-GPU
        cmd = [
            "deepspeed",
            "--num_gpus", str(num_gpus),
            "--master_port", "29500",
            str(llamafactory_path / "src" / "train.py"),
            str(config_path),
            "--dataset_dir", str(workspace / "data"),
            "--dataset_info", str(dataset_info),
        ]
    else:
        # Single GPU or no DeepSpeed
        cmd = [
            "python",
            str(llamafactory_path / "src" / "train.py"),
            str(config_path),
            "--dataset_dir", str(workspace / "data"),
            "--dataset_info", str(dataset_info),
        ]
    
    if resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint])
    
    print("\n" + "="*80)
    print("Starting training with command:")
    print(" ".join(cmd))
    print("="*80 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True, cwd=workspace)
        print("\n✓ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL on click location dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (8 GPUs, DeepSpeed)
  python scripts/train.py
  
  # Train with custom config
  python scripts/train.py --config configs/custom_config.yaml
  
  # Resume from checkpoint
  python scripts/train.py --resume saves/qwen3-vl-8b/lora/sft/checkpoint-1000
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_vl_lora.yaml",
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--dataset-info",
        type=str,
        default="configs/dataset_info.json",
        help="Path to dataset_info.json"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--no-deepspeed",
        action="store_true",
        help="Disable DeepSpeed (not recommended for multi-GPU)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency checks"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Qwen3-VL Fine-tuning Training Script")
    print("="*80)
    
    # Setup
    setup_environment()
    
    # Check dependencies
    if not args.skip_checks:
        print("\nChecking dependencies...")
        if not check_dependencies():
            print("\n✗ Dependency check failed!")
            sys.exit(1)
    
    # Verify config and data files exist
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"\n✗ Config file not found: {config_path}")
        sys.exit(1)
    
    dataset_info_path = Path(args.dataset_info)
    if not dataset_info_path.exists():
        print(f"\n✗ Dataset info file not found: {dataset_info_path}")
        sys.exit(1)
    
    # Run training
    print("\n" + "="*80)
    print("Launching training...")
    print("="*80 + "\n")
    
    run_training(
        config_path=str(config_path),
        dataset_info=str(dataset_info_path),
        num_gpus=args.num_gpus,
        use_deepspeed=not args.no_deepspeed,
        resume_from_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()

