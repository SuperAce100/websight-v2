#!/usr/bin/env python3
"""
Quick test script for the fine-tuned Qwen3-VL model.
Usage: python test_model.py --image path/to/image.png --prompt "click the button"
"""

import argparse
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image


def load_model(adapter_path: str = None, model_path: str = None, base_model: str = "Qwen/Qwen3-VL-8B-Instruct"):
    """
    Load the model - either with LoRA adapter or as merged model.
    
    Args:
        adapter_path: Path to LoRA adapter (if using adapter)
        model_path: Path to merged model (if using merged model)
        base_model: Base model name (only used with adapter_path)
    """
    from pathlib import Path
    
    if model_path:
        # Load merged model directly
        print(f"Loading merged model: {model_path}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    elif adapter_path:
        # Load base model with LoRA adapter
        print(f"Loading base model: {base_model}")
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        raise ValueError("Either --adapter-path or --model-path must be provided")
    
    model.eval()
    return model, processor


def predict_click(model, processor, image_path: str, prompt: str):
    """Generate click coordinates for the given image and prompt."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare input
    text = f"<image>\n{prompt}"
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    print(f"\nPrompt: {prompt}")
    print("Generating click coordinates...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Deterministic output
            temperature=None,
        )
    
    # Decode
    result = processor.decode(outputs[0], skip_special_tokens=True)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Qwen3-VL model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (use with --base-model)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to merged model (alternative to --adapter-path)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model name or path (only used with --adapter-path)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Click instruction (e.g., 'click the login button')",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.adapter_path and not args.model_path:
        parser.error("Either --adapter-path or --model-path must be provided")
    
    if args.adapter_path and args.model_path:
        parser.error("Cannot use both --adapter-path and --model-path. Choose one.")
    
    # Load model
    model, processor = load_model(
        adapter_path=args.adapter_path,
        model_path=args.model_path,
        base_model=args.base_model
    )
    
    # Generate prediction
    result = predict_click(model, processor, args.image, args.prompt)
    
    print(f"\n{'='*50}")
    print(f"Result: {result}")
    print(f"{'='*50}\n")
    
    # Try to extract coordinates
    if "pyautogui.click" in result:
        import re
        match = re.search(r"pyautogui\.click\((\d+),\s*(\d+)\)", result)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            print(f"✓ Extracted coordinates: x={x}, y={y}")
            print(f"  (normalized to 1400x800 resolution)")
        else:
            print("⚠ Could not parse coordinates from output")
    else:
        print("⚠ Output doesn't match expected format")


if __name__ == "__main__":
    main()

