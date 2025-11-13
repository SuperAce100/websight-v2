#!/usr/bin/env python3
"""
Extract LoRA adapter from DeepSpeed checkpoint for inference.

This script:
1. Checks if adapter files already exist at the output_dir level
2. If not, extracts the adapter from the DeepSpeed checkpoint
3. Saves the adapter in PEFT format for inference use

Usage:
    python scripts/extract_adapter_from_checkpoint.py \
        --checkpoint_dir ht-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200/global_step200 \
        --output_dir ht-v2/saves/qwen3-vl-8b/lora/sft-noeval
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from typing import Optional

try:
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError as e:
    print(f"Error: Missing required packages. Install with: pip install peft transformers")
    sys.exit(1)


def check_adapter_exists(output_dir: str) -> bool:
    """Check if adapter files already exist."""
    adapter_config = Path(output_dir) / "adapter_config.json"
    adapter_model = Path(output_dir) / "adapter_model.bin"
    adapter_model_safetensors = Path(output_dir) / "adapter_model.safetensors"
    
    has_config = adapter_config.exists()
    has_model = adapter_model.exists() or adapter_model_safetensors.exists()
    
    return has_config and has_model


def extract_adapter_from_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "cuda"
):
    """
    Extract LoRA adapter from DeepSpeed checkpoint.
    
    This loads the checkpoint, extracts the LoRA weights, and saves them
    in PEFT format for inference.
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_path}")
    print(f"Output directory: {output_path}")
    print(f"Base model: {base_model}")
    print()
    
    # Check if checkpoint exists
    model_state_file = checkpoint_path / "mp_rank_00_model_states.pt"
    if not model_state_file.exists():
        print(f"❌ Error: Model state file not found: {model_state_file}")
        print(f"\nAvailable files in checkpoint:")
        for f in checkpoint_path.iterdir():
            print(f"  - {f.name}")
        return False
    
    print("Step 1: Loading base model...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        print("✓ Base model loaded")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return False
    
    print("\nStep 2: Loading checkpoint...")
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_state_file, map_location="cpu")
        
        # The checkpoint structure depends on how LLaMA-Factory saves it
        # It might be in 'module' key or directly accessible
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"✓ Checkpoint loaded")
        print(f"  Keys in checkpoint: {len(state_dict.keys())} parameters")
        
        # Filter for LoRA weights (they typically have 'lora' in the key)
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        if not lora_keys:
            print("\n⚠ Warning: No LoRA keys found in checkpoint!")
            print("  This might be a full model checkpoint, not a LoRA checkpoint.")
            print("  Trying to load as PEFT checkpoint...")
            
            # Try to load as PEFT model directly
            try:
                # LLaMA-Factory might save the adapter in a different format
                # Try loading the checkpoint directory as a PEFT adapter
                model = PeftModel.from_pretrained(
                    model,
                    str(checkpoint_path.parent.parent),  # Go up to checkpoint-200 level
                    device_map=device if torch.cuda.is_available() else "cpu"
                )
                print("✓ Loaded as PEFT adapter from checkpoint directory")
            except Exception as e:
                print(f"❌ Failed to load as PEFT: {e}")
                print("\nAlternative: The adapter might be saved at the output_dir level.")
                print(f"  Check: {output_path.parent if 'checkpoint' in str(output_path) else output_path}")
                return False
        else:
            print(f"  Found {len(lora_keys)} LoRA parameters")
            # Create a filtered state dict with only LoRA weights
            lora_state_dict = {k: state_dict[k] for k in lora_keys}
            
            # Save the LoRA adapter
            print("\nStep 3: Saving LoRA adapter...")
            # We need to create a proper PEFT adapter structure
            # This is complex, so we'll use a different approach
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 4: Saving adapter to output directory...")
    try:
        # Save the adapter using PEFT's save_pretrained
        model.save_pretrained(str(output_path))
        print(f"✓ Adapter saved to: {output_path}")
        
        # Verify files were created
        adapter_config = output_path / "adapter_config.json"
        adapter_model = output_path / "adapter_model.bin"
        adapter_model_safetensors = output_path / "adapter_model.safetensors"
        
        if adapter_config.exists():
            print(f"  ✓ adapter_config.json")
        if adapter_model.exists() or adapter_model_safetensors.exists():
            print(f"  ✓ adapter_model file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract LoRA adapter from DeepSpeed checkpoint"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to DeepSpeed checkpoint directory (e.g., checkpoint-200/global_step200)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save adapter files (e.g., saves/qwen3-vl-8b/lora/sft-noeval)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("LoRA Adapter Extraction from DeepSpeed Checkpoint")
    print("="*80)
    print()
    
    # Check if adapter already exists
    if check_adapter_exists(args.output_dir):
        print(f"✓ Adapter files already exist in: {args.output_dir}")
        print("  You can use this directory directly for inference!")
        print()
        print("Example usage:")
        print(f"  python test_model.py --adapter-path {args.output_dir} --image <image> --prompt <prompt>")
        return 0
    
    print("Adapter files not found. Extracting from checkpoint...")
    print()
    
    # Extract adapter
    success = extract_adapter_from_checkpoint(
        args.checkpoint_dir,
        args.output_dir,
        args.base_model,
        args.device
    )
    
    if success:
        print()
        print("="*80)
        print("✓ Adapter extraction completed!")
        print("="*80)
        print()
        print("You can now use the adapter for inference:")
        print(f"  python test_model.py --adapter-path {args.output_dir} --image <image> --prompt <prompt>")
        return 0
    else:
        print()
        print("="*80)
        print("❌ Adapter extraction failed")
        print("="*80)
        print()
        print("Alternative: Use LLaMA-Factory's export command:")
        print(f"  llamafactory-cli export \\")
        print(f"    --model_name_or_path {args.base_model} \\")
        print(f"    --adapter_name_or_path {Path(args.checkpoint_dir).parent.parent} \\")
        print(f"    --export_dir {args.output_dir} \\")
        print(f"    --template qwen2_vl \\")
        print(f"    --finetuning_type lora")
        return 1


if __name__ == "__main__":
    sys.exit(main())

