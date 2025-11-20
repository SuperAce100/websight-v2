#!/usr/bin/env python3
"""
Run pure inference on Qwen3-VL-8B model for waveUI test dataset.

This script performs inference-only evaluation:
1. Loads the trained model from HuggingFace (supports both merged models and LoRA adapters)
2. Loads the test dataset in the same format as training (ShareGPT format with messages)
3. Runs inference on each sample with torch.no_grad() (no gradients computed)
4. Uses the same input format as training (qwen2_vl template applied automatically by processor)
5. Saves model predictions (click locations) to JSONL file

The input format matches training exactly:
- Messages array with "system" role (system prompt) and "user" role containing "<image>\n{prompt}"
- Images referenced in the "images" field
- Processor automatically applies qwen2_vl template (same as training config)

Predictions are saved to the specified output file (default: inference_results.jsonl).
Each prediction contains the model's output text (e.g., "pyautogui.click(x, y)").
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
)
from PIL import Image


def load_model(model_name_or_path: str, adapter_path: str = None, device: str = "cuda"):
    """
    Load the model and processor for inference.
    Supports both merged models and LoRA adapters.
    
    The processor automatically applies the qwen2_vl template (same as training)
    to ensure input format matches training exactly.

    Args:
        model_name_or_path: Path to model (HuggingFace model ID or local path)
        adapter_path: Optional path to LoRA adapter (if using LoRA)
        device: Device to load model on

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model from {model_name_or_path}...")
    print("  Mode: INFERENCE ONLY (no gradients will be computed)")
    
    # Load processor (automatically applies qwen2_vl template for Qwen3-VL)
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    print("  Processor loaded (qwen2_vl template applied automatically)")
    
    # Load base model
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapter if provided
    if adapter_path:
        try:
            from peft import PeftModel
            print(f"Loading LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map=device
            )
        except ImportError:
            print("Warning: peft not installed, skipping LoRA adapter loading")
        except Exception as e:
            print(f"Warning: Failed to load LoRA adapter: {e}")
    
    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    print("  Model loaded successfully and set to eval() mode")
    
    return model, processor


def load_test_dataset(test_file: str) -> List[Dict]:
    """
    Load test dataset from JSONL file.

    Args:
        test_file: Path to test JSONL file

    Returns:
        List of test records
    """
    print(f"Loading test dataset from {test_file}...")
    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} test records")
    return records


def run_inference(
    model,
    processor,
    test_records: List[Dict],
    media_dir: str,
    device: str = "cuda",
    output_file: str = "inference_results.jsonl",
) -> None:
    """
    Run pure inference on test dataset and save predictions.
    
    This function runs inference with NO gradient computation (torch.no_grad()).
    Input format matches training exactly:
    - Uses the same message format: [{"role": "system", ...}, {"role": "user", "content": "<image>\\n{prompt}"}]
    - Includes system prompt (same as training)
    - Processor automatically applies qwen2_vl template (same as training)
    - Images are loaded from media_dir + image_path

    Args:
        model: Loaded model (in eval mode)
        processor: Loaded processor (applies qwen2_vl template automatically)
        test_records: List of test records in ShareGPT format (same as training)
        media_dir: Base directory for images
        device: Device to run inference on
        output_file: Path to save predictions (JSONL format)
        
    Output format (JSONL):
        Each line contains:
        {
            "input": {
                "prompt": "<image>\\n{prompt_text}",
                "image_path": "relative/path/to/image.png"
            },
            "output": "pyautogui.click(x, y)"  # Model prediction
        }
    """
    print(f"Running INFERENCE (no gradients) on {len(test_records)} samples...")
    print(f"  Input format: Same as training (system prompt + user message, qwen2_vl template)")
    print(f"  Output file: {output_file}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(test_records, desc="Running inference"):
            # Extract messages and image (same format as training - includes system prompt)
            messages = record.get("messages", [])
            images = record.get("images", [])
            
            if not messages or not images:
                failed += 1
                continue
            
            image_path = images[0]
            
            # Find user message for saving in output (for reference)
            user_message = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message or not image_path:
                failed += 1
                continue
            
            # Load image
            full_image_path = os.path.join(media_dir, image_path)
            if not os.path.exists(full_image_path):
                print(f"Warning: Image not found: {full_image_path}")
                failed += 1
                continue
            
            try:
                image = Image.open(full_image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load image {full_image_path}: {e}")
                failed += 1
                continue
            
            # Prepare inputs (processor applies qwen2_vl template automatically)
            # Pass full messages array (system + user) to match training format
            try:
                # Try using messages parameter first (for conversation format)
                # If that fails, fall back to text parameter
                try:
                    inputs = processor(
                        messages=messages,  # Full messages array (system + user) - same as training
                        images=image,
                        return_tensors="pt"
                    ).to(device)
                except TypeError:
                    # Fallback: some processors use 'text' parameter for messages
                    inputs = processor(
                        text=messages,  # Full messages array (system + user) - same as training
                        images=image,
                        return_tensors="pt"
                    ).to(device)
            except Exception as e:
                print(f"Warning: Failed to process inputs: {e}")
                failed += 1
                continue
            
            # Generate prediction with NO GRADIENTS (pure inference)
            with torch.no_grad():  # Disables gradient computation
                try:
                    # Get input length to extract only generated tokens
                    input_length = inputs.input_ids.shape[1]
                    
                    # Generate with deterministic decoding (same as training inference)
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,  # Deterministic (greedy decoding)
                    )
                    
                    # Extract only the generated tokens (excluding input)
                    generated_ids_trimmed = generated_ids[0][input_length:]
                    
                    # Decode only the generated part
                    output_text = processor.decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ).strip()
                    
                    # Save prediction (click location)
                    result = {
                        "input": {
                            "prompt": user_message,
                            "image_path": image_path,
                        },
                        "output": output_text,  # Model prediction (e.g., "pyautogui.click(x, y)")
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    successful += 1
                    
                except Exception as e:
                    print(f"Warning: Generation failed: {e}")
                    failed += 1
                    continue
    
    print(f"\n✓ Inference complete (no gradients computed)!")
    print(f"  Successful predictions: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run pure inference (no gradients) on Qwen3-VL-8B model for waveUI test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs inference-only evaluation with NO gradient computation.
The input format matches training exactly (qwen2_vl template applied automatically).

Predictions (click locations) are saved to the specified output file in JSONL format.
Each line contains the model's output (e.g., "pyautogui.click(x, y)").

Example:
  python scripts/test_after_grounding.py \\
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \\
    --test-file data/wave_ui_test.jsonl \\
    --output predictions.jsonl
        """
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="HuggingFace model ID or path to trained model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional path to LoRA adapter (if using LoRA instead of merged model)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/wave_ui_test.jsonl",
        help="Path to test JSONL file (ShareGPT format with 'messages' and 'images', same as training)",
    )
    parser.add_argument(
        "--media-dir",
        type=str,
        default="wave-ui",
        help="Base directory for images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results.jsonl",
        help="Path to save predictions (JSONL format). Each line contains model output (click location).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)",
    )
    
    args = parser.parse_args()
    
    # Check if test file exists
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        print("Please run scripts/prepare_test_split.py first to create the test split")
        return 1
    
    print("="*80)
    print("Qwen3-VL Inference")
    print("="*80)
    print(f"Test file: {args.test_file}")
    print(f"Output file: {args.output}")
    print(f"Media directory: {args.media_dir}")
    print("="*80)
    print()
    
    # Load model (set to eval mode, no gradients)
    model, processor = load_model(args.model_name_or_path, args.adapter_path, args.device)
    
    # Load test dataset (same format as training)
    test_records = load_test_dataset(args.test_file)
    
    if len(test_records) == 0:
        print("Error: No test records found!")
        return 1
    
    # Run inference (no gradients, pure inference)
    run_inference(
        model,
        processor,
        test_records,
        args.media_dir,
        args.device,
        args.output,
    )
    
    print()
    print("="*80)
    print(f"✓ Predictions saved to: {args.output}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
