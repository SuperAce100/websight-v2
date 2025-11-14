#!/usr/bin/env python3
"""
Test script to verify ScreenSpot-Pro benchmark setup.

This script checks that all required dependencies and scripts are available.

Usage:
    python scripts/test_screenspot_setup.py
"""

import sys
from pathlib import Path


def check_file(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} (missing: {path})")
        return False


def check_import(module: str, package: str = None) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module)
        display_name = package or module
        print(f"  ✓ {display_name}")
        return True
    except ImportError:
        display_name = package or module
        print(f"  ✗ {display_name} (install with: pip install {package or module})")
        return False


def main():
    print("=" * 80)
    print("ScreenSpot-Pro Benchmark Setup Test")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # Check scripts
    print("Checking scripts...")
    all_ok &= check_file("scripts/screenspot_pro_utils.py", "Dataset utilities")
    all_ok &= check_file("scripts/prepare_screenspot_pro.py", "Dataset preparation script")
    all_ok &= check_file("scripts/run_screenspot_benchmark.py", "Benchmark runner script")
    all_ok &= check_file("scripts/evaluate_screenspot.py", "Evaluation script")
    all_ok &= check_file("slurm/screenspot_benchmark.slurm", "Slurm job script")
    print()
    
    # Check documentation
    print("Checking documentation...")
    all_ok &= check_file("SCREENSPOT_BENCHMARK.md", "Main documentation")
    all_ok &= check_file("SCREENSPOT_QUICKSTART.md", "Quick reference")
    print()
    
    # Check Python dependencies
    print("Checking Python dependencies...")
    all_ok &= check_import("torch", "torch")
    all_ok &= check_import("transformers", "transformers")
    all_ok &= check_import("PIL", "pillow")
    all_ok &= check_import("tqdm", "tqdm")
    all_ok &= check_import("huggingface_hub", "huggingface-hub")
    print()
    
    # Check optional dependencies
    print("Checking optional dependencies...")
    peft_ok = check_import("peft", "peft")
    if not peft_ok:
        print("    Note: peft is optional, only needed for LoRA adapters")
    print()
    
    # Summary
    print("=" * 80)
    if all_ok:
        print("✓ All required components are available!")
        print()
        print("Next steps:")
        print("  1. Prepare dataset:")
        print("     python scripts/prepare_screenspot_pro.py")
        print()
        print("  2. Run benchmark:")
        print("     python scripts/run_screenspot_benchmark.py \\")
        print("       --model-name-or-path Asanshay/websight-v2-grounded")
        print()
        print("  3. Or submit Slurm job:")
        print("     sbatch slurm/screenspot_benchmark.slurm")
        print()
        return 0
    else:
        print("✗ Some components are missing. Please install missing dependencies.")
        print()
        print("Install all dependencies:")
        print("  pip install torch transformers pillow tqdm huggingface-hub")
        print()
        print("Optional (for LoRA adapters):")
        print("  pip install peft")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

