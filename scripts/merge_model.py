#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for download and HuggingFace upload.

This script:
1. Merges the LoRA adapter with the base model
2. Saves the merged model in standard HuggingFace format
3. Optionally pushes to HuggingFace Hub
4. Creates a model card (README.md)

Usage:
    # Just merge (for download)
    python scripts/merge_model.py \
        --adapter_path /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200 \
        --output_dir /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged \
        --base_model Qwen/Qwen3-VL-8B-Instruct

    # Merge and push to HuggingFace
    python scripts/merge_model.py \
        --adapter_path /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b/lora/sft-noeval/checkpoint-200 \
        --output_dir /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b-merged \
        --base_model Qwen/Qwen3-VL-8B-Instruct \
        --push_to_hub \
        --hub_model_id your-username/qwen3-vl-8b-websight \
        --hub_token YOUR_HF_TOKEN
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    import torch
except ImportError as e:
    print(f"Error: Missing required packages. Install with: pip install transformers peft")
    sys.exit(1)

try:
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Cannot push to HuggingFace.")


def create_model_card(output_dir: Path, base_model: str, hub_model_id: Optional[str] = None) -> None:
    """Create a model card (README.md) for the merged model."""
    readme_content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - qwen3-vl
  - vision
  - gui-automation
  - websight
  - fine-tuned
datasets:
  - wave-ui/websight-v2
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3-VL-8B WebSight Fine-tuned

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) on the WebSight dataset for GUI automation tasks.

## Model Description

- **Base Model**: {base_model}
- **Fine-tuning Method**: LoRA (merged)
- **Dataset**: wave-ui/websight-v2
- **Task**: Image-to-click location prediction
- **Output Format**: `pyautogui.click(x, y)` commands

## Usage

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    "{hub_model_id if hub_model_id else 'REPO_NAME'}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "{hub_model_id if hub_model_id else 'REPO_NAME'}",
    trust_remote_code=True
)

# Prepare input
image = Image.open("screenshot.png")
prompt = "click the login button"

inputs = processor(
    text=f"<image>\\n{{prompt}}",
    images=image,
    return_tensors="pt"
).to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)  # Output: pyautogui.click(x, y)
```

## Training Details

- **Training Framework**: LLaMA-Factory
- **Hardware**: 8x H100 GPUs
- **LoRA Config**:
  - Rank: 64
  - Alpha: 128
  - Dropout: 0.05
  - Target modules: all linear layers

## Output Format

The model outputs click coordinates normalized to 1400x800 resolution:
- Format: `pyautogui.click(x, y)`
- Example: `pyautogui.click(565, 486)`

Scale to your screen resolution:
```python
x_actual = int(x_norm * (screen_width / 1400))
y_actual = int(y_norm * (screen_height / 800))
```

## Citation

```bibtex
@misc{{qwen3-vl-websight,
  title={{Qwen3-VL Fine-tuned for GUI Automation}},
  author={{Your Name}},
  year={{2025}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/{hub_model_id if hub_model_id else 'REPO_NAME'}}}}}
}}
```

## License

Apache 2.0 (inherited from base model)
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Created model card: {readme_path}")


def merge_model(
    adapter_path: str,
    output_dir: str,
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "auto",
    use_llamafactory: bool = True
) -> bool:
    """
    Merge LoRA adapter with base model.
    
    Args:
        adapter_path: Path to LoRA adapter directory
        output_dir: Output directory for merged model
        base_model: Base model name or path
        device: Device to use for merging
        use_llamafactory: If True, use llamafactory-cli (faster). If False, use Python API.
    
    Returns:
        True if successful, False otherwise
    """
    adapter_path_obj = Path(adapter_path)
    output_path_obj = Path(output_dir)
    
    # Check adapter exists
    if not adapter_path_obj.exists():
        print(f"❌ Error: Adapter path not found: {adapter_path}")
        return False
    
    adapter_config = adapter_path_obj / "adapter_config.json"
    if not adapter_config.exists():
        print(f"❌ Error: adapter_config.json not found in {adapter_path}")
        return False
    
    print(f"Adapter path: {adapter_path}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {base_model}")
    print()
    
    # Use LLaMA-Factory if available (faster and more reliable)
    if use_llamafactory:
        try:
            import subprocess
            print("Using LLaMA-Factory to merge model...")
            print("This may take 10-30 minutes depending on model size...")
            
            # Create output directory
            output_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Run llamafactory-cli export
            cmd = [
                "llamafactory-cli", "export",
                "--model_name_or_path", base_model,
                "--adapter_name_or_path", str(adapter_path_obj),
                "--export_dir", str(output_path_obj),
                "--template", "qwen2_vl",
                "--finetuning_type", "lora",
                "--export_size", "2",
                "--export_legacy_format", "False"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print("✓ Model merged successfully using LLaMA-Factory")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ LLaMA-Factory merge failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            print("\nFalling back to Python API method...")
            use_llamafactory = False
        except FileNotFoundError:
            print("⚠ llamafactory-cli not found. Falling back to Python API method...")
            use_llamafactory = False
    
    # Fallback: Use Python API (slower but more compatible)
    if not use_llamafactory:
        print("Using Python API to merge model...")
        print("This may take 20-40 minutes depending on model size...")
        
        try:
            # Load base model
            print("Step 1: Loading base model...")
            model = AutoModelForVision2Seq.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
            print("✓ Base model loaded")
            
            # Load LoRA adapter
            print("\nStep 2: Loading LoRA adapter...")
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path_obj),
                device_map=device
            )
            print("✓ LoRA adapter loaded")
            
            # Merge adapter into base model
            print("\nStep 3: Merging adapter into base model...")
            model = model.merge_and_unload()
            print("✓ Adapter merged")
            
            # Save merged model
            print("\nStep 4: Saving merged model...")
            output_path_obj.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(
                str(output_path_obj),
                safe_serialization=True,
                max_shard_size="2GB"
            )
            
            # Save processor/tokenizer
            print("Step 5: Saving processor and tokenizer...")
            processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            processor.save_pretrained(str(output_path_obj))
            
            print("✓ Merged model saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error merging model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def push_to_hub(
    model_dir: str,
    hub_model_id: str,
    hub_token: Optional[str] = None,
    private: bool = False
) -> bool:
    """Push merged model to HuggingFace Hub."""
    if not HF_AVAILABLE:
        print("❌ Error: huggingface_hub not available. Install with: pip install huggingface_hub")
        return False
    
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        return False
    
    print(f"\nPushing model to HuggingFace Hub...")
    print(f"  Repository: {hub_model_id}")
    print(f"  Private: {private}")
    
    try:
        # Login if token provided
        if hub_token:
            login(token=hub_token, add_to_git_credential=True)
            print("✓ Logged in to HuggingFace")
        
        # Create repository
        api = HfApi()
        try:
            api.create_repo(
                repo_id=hub_model_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            print(f"✓ Repository created/verified")
        except Exception as e:
            print(f"⚠ Repository creation: {e}")
        
        # Upload model
        print(f"\nUploading model files...")
        print(f"This may take 10-30 minutes depending on your connection...")
        
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=hub_model_id,
            repo_type="model",
            commit_message="Upload merged Qwen3-VL-8B WebSight fine-tuned model"
        )
        
        print(f"\n✓ Model pushed successfully!")
        print(f"  View at: https://huggingface.co/{hub_model_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing to HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for download/HuggingFace upload"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (e.g., checkpoint-200)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model"
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
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--use_llamafactory",
        action="store_true",
        default=True,
        help="Use llamafactory-cli for merging (faster, default: True)"
    )
    parser.add_argument(
        "--no_llamafactory",
        action="store_false",
        dest="use_llamafactory",
        help="Use Python API instead of llamafactory-cli"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="HuggingFace model ID (e.g., username/model-name)"
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace repository private"
    )
    parser.add_argument(
        "--skip_model_card",
        action="store_true",
        help="Skip creating model card (README.md)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Merge LoRA Adapter with Base Model")
    print("="*80)
    print()
    
    # Merge model
    success = merge_model(
        args.adapter_path,
        args.output_dir,
        args.base_model,
        args.device,
        args.use_llamafactory
    )
    
    if not success:
        print("\n❌ Model merge failed!")
        return 1
    
    # Create model card
    if not args.skip_model_card:
        print("\nCreating model card...")
        create_model_card(
            Path(args.output_dir),
            args.base_model,
            args.hub_model_id
        )
    
    # Check model size
    output_path = Path(args.output_dir)
    if output_path.exists():
        import subprocess
        try:
            result = subprocess.run(
                ["du", "-sh", str(output_path)],
                capture_output=True,
                text=True,
                check=True
            )
            size = result.stdout.split()[0]
            print(f"\n✓ Merged model size: {size}")
        except:
            pass
    
    # Push to HuggingFace if requested
    if args.push_to_hub:
        if not args.hub_model_id:
            print("\n❌ Error: --hub_model_id required when using --push_to_hub")
            return 1
        
        hub_token = args.hub_token or os.environ.get("HF_TOKEN")
        if not hub_token:
            print("\n⚠ Warning: No HuggingFace token provided.")
            print("  Set --hub_token or HF_TOKEN environment variable")
            print("  Skipping push to Hub...")
        else:
            success = push_to_hub(
                args.output_dir,
                args.hub_model_id,
                hub_token,
                args.private
            )
            if not success:
                return 1
    
    print()
    print("="*80)
    print("✓ Merge completed successfully!")
    print("="*80)
    print()
    print(f"Merged model saved to: {args.output_dir}")
    print()
    print("Next steps:")
    print("  1. Download the merged model to your local computer:")
    print(f"     rsync -avz --progress {args.output_dir}/ local/path/")
    print()
    print("  2. Or use it directly for inference:")
    print(f"     python test_model.py --model-path {args.output_dir} --image <image> --prompt <prompt>")
    print()
    if args.push_to_hub and args.hub_model_id:
        print(f"  3. Model is available on HuggingFace:")
        print(f"     https://huggingface.co/{args.hub_model_id}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

