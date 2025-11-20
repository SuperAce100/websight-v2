#!/usr/bin/env python3
"""
Extract LoRA adapter weights directly from DeepSpeed checkpoint files.

This script loads the checkpoint state dict, extracts LoRA weights,
and saves them in PEFT format for inference.

Usage:
    python scripts/extract_lora_from_checkpoint.py \
        --checkpoint_file ht-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200/global_step200/mp_rank_00_model_states.pt \
        --output_dir ht-v2/saves/qwen3-vl-8b/lora/sft-noeval-adapter \
        --base_model Qwen/Qwen3-VL-8B-Instruct \
        --lora_rank 64 \
        --lora_alpha 128
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from typing import Dict, Any

try:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForVision2Seq
except ImportError as e:
    print(f"Error: Missing required packages. Install with: pip install peft transformers")
    sys.exit(1)


def inspect_checkpoint(checkpoint_file: str):
    """Inspect the structure of the checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    
    print("\nCheckpoint structure:")
    print(f"  Top-level keys: {list(checkpoint.keys())}")
    
    # Try to find the model state dict
    state_dict = None
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
        print("  Found 'module' key")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("  Found 'model' key")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("  Found 'state_dict' key")
    else:
        # Check if it's directly a state dict
        if isinstance(checkpoint, dict) and any('weight' in str(v) or 'bias' in str(v) for v in checkpoint.values()):
            state_dict = checkpoint
            print("  Checkpoint appears to be a state dict directly")
    
    if state_dict is None:
        print("\n⚠ Could not identify model state dict structure")
        print("  Available keys:", list(checkpoint.keys())[:10])
        return None
    
    print(f"\nState dict contains {len(state_dict)} parameters")
    
    # Look for LoRA weights
    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
    print(f"\nLoRA-related keys found: {len(lora_keys)}")
    if lora_keys:
        print("  Sample LoRA keys:")
        for key in lora_keys[:5]:
            print(f"    - {key}")
        if len(lora_keys) > 5:
            print(f"    ... and {len(lora_keys) - 5} more")
    
    return state_dict, lora_keys


def extract_lora_weights(state_dict: Dict[str, torch.Tensor], lora_keys: list) -> Dict[str, torch.Tensor]:
    """Extract LoRA weights from the full state dict."""
    lora_state_dict = {}
    
    for key in lora_keys:
        lora_state_dict[key] = state_dict[key].clone()
    
    return lora_state_dict


def create_adapter_config(
    base_model: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float = 0.05,
    target_modules: list = None
) -> Dict[str, Any]:
    """Create adapter_config.json content."""
    if target_modules is None:
        # Default target modules for Qwen models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "target_modules": target_modules,
        "task_type": "FEATURE_EXTRACTION"
    }
    
    return config


def save_adapter(
    lora_state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    adapter_config: Dict[str, Any],
    use_safetensors: bool = True
):
    """Save LoRA adapter in PEFT format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save adapter config
    config_path = output_path / "adapter_config.json"
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    print(f"✓ Saved adapter_config.json to {config_path}")
    
    # Save adapter weights
    if use_safetensors:
        try:
            from safetensors.torch import save_file
            # Convert state dict keys to match PEFT format if needed
            # PEFT expects keys like "base_model.model.layer.0.attention.self.query.lora_A.weight"
            # But checkpoint might have different format
            
            # Try to save directly first
            model_path = output_path / "adapter_model.safetensors"
            save_file(lora_state_dict, model_path)
            print(f"✓ Saved adapter_model.safetensors to {model_path}")
        except ImportError:
            print("⚠ safetensors not available, using .bin format")
            use_safetensors = False
    
    if not use_safetensors:
        model_path = output_path / "adapter_model.bin"
        torch.save(lora_state_dict, model_path)
        print(f"✓ Saved adapter_model.bin to {model_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract LoRA adapter from DeepSpeed checkpoint"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="Path to mp_rank_00_model_states.pt file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save adapter files"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank (should match training config)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha (should match training config)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect checkpoint structure, don't extract"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Extract LoRA Adapter from DeepSpeed Checkpoint")
    print("="*80)
    print()
    
    # Check if checkpoint file exists
    checkpoint_path = Path(args.checkpoint_file)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    # Inspect checkpoint
    result = inspect_checkpoint(str(checkpoint_path))
    if result is None:
        return 1
    
    state_dict, lora_keys = result
    
    if args.inspect_only:
        print("\n✓ Inspection complete. Use without --inspect_only to extract adapter.")
        return 0
    
    if not lora_keys:
        print("\n❌ Error: No LoRA weights found in checkpoint!")
        print("\nPossible reasons:")
        print("  1. This might not be a LoRA checkpoint")
        print("  2. LoRA weights might be stored differently")
        print("  3. Try using LLaMA-Factory's export command instead:")
        print(f"     llamafactory-cli export \\")
        print(f"       --model_name_or_path {args.base_model} \\")
        print(f"       --adapter_name_or_path {checkpoint_path.parent.parent} \\")
        print(f"       --export_dir {args.output_dir} \\")
        print(f"       --template qwen2_vl \\")
        print(f"       --finetuning_type lora")
        return 1
    
    print(f"\nExtracting {len(lora_keys)} LoRA parameters...")
    
    # Extract LoRA weights
    lora_state_dict = extract_lora_weights(state_dict, lora_keys)
    
    # Create adapter config
    adapter_config = create_adapter_config(
        args.base_model,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout
    )
    
    # Save adapter
    output_path = save_adapter(
        lora_state_dict,
        args.output_dir,
        adapter_config
    )
    
    print()
    print("="*80)
    print("✓ Adapter extraction completed!")
    print("="*80)
    print()
    print(f"Adapter saved to: {output_path}")
    print()
    print("You can now use it for inference:")
    print(f"  python test_model.py --adapter-path {output_path} --image <image> --prompt <prompt>")
    print()
    print("Or in Python:")
    print("  from peft import PeftModel")
    print("  from transformers import AutoModelForVision2Seq")
    print(f"  model = AutoModelForVision2Seq.from_pretrained('{args.base_model}')")
    print(f"  model = PeftModel.from_pretrained(model, '{output_path}')")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

