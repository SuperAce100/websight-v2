#!/usr/bin/env python3
"""
Run ScreenSpot-Pro benchmark using HuggingFace models.

This script loads a vision-language model from HuggingFace and runs inference
on the ScreenSpot-Pro dataset, saving predictions for later evaluation.

Usage:
    python scripts/run_screenspot_benchmark.py \\
        --model-name-or-path Asanshay/websight-v2-grounded \\
        --data-dir screenspot_pro \\
        --output runs/screenspot_pro/predictions.jsonl
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from screenspot_pro_utils import load_screenspot_pro


def get_images_dir_from_records(records: List[Dict], data_dir: str) -> str:
    """
    Determine images directory from records or data_dir.
    
    Args:
        records: List of records
        data_dir: Data directory path
    
    Returns:
        Images directory path
    """
    # Check if records have absolute image paths
    if records and "image_path" in records[0]:
        first_image = records[0]["image_path"]
        # If it starts with "images/", it's relative to data_dir
        if first_image.startswith("images/"):
            return data_dir
    
    return data_dir


def parse_coordinates(text: str) -> Optional[Tuple[int, int]]:
    """
    Parse coordinates from model output.
    
    Args:
        text: Model output text
    
    Returns:
        Tuple of (x, y) coordinates if found, None otherwise
    """
    # Pattern to match pyautogui.click(x, y)
    pattern = r'pyautogui\.click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    match = re.search(pattern, text)
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x, y)
    
    # Fallback: match any coordinates
    pattern = r'\(?(\d+)\s*,\s*(\d+)\)?'
    match = re.search(pattern, text)
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x, y)
    
    return None


def point_in_bbox(x: float, y: float, bbox: List[float]) -> bool:
    """
    Check if a point is within a bounding box.
    
    Args:
        x: X coordinate
        y: Y coordinate
        bbox: [x_min, y_min, x_max, y_max]
    
    Returns:
        True if point is within bbox, False otherwise
    """
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def load_model(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda"
):
    """
    Load model and processor for inference.
    
    Args:
        model_name_or_path: HuggingFace model ID or local path
        adapter_path: Optional path to LoRA adapter
        device: Device to load model on
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model from {model_name_or_path}...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    print("  ✓ Processor loaded")
    
    # Load model
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
            print("  ✓ LoRA adapter loaded")
        except ImportError:
            print("Warning: peft not installed, skipping LoRA adapter")
        except Exception as e:
            print(f"Warning: Failed to load LoRA adapter: {e}")
    
    # Set to evaluation mode
    model.eval()
    print("  ✓ Model loaded and set to eval mode")
    
    return model, processor


def run_inference(
    model,
    processor,
    records: List[Dict],
    media_dir: str,
    device: str,
    output_file: str,
    max_new_tokens: int = 512,
    limit: Optional[int] = None,
    progress_interval: int = 50
) -> None:
    """
    Run inference on ScreenSpot-Pro records and save predictions.
    
    Args:
        model: Loaded model in eval mode
        processor: Loaded processor
        records: List of test records
        media_dir: Base directory for images
        device: Device to run inference on
        output_file: Path to save predictions
        max_new_tokens: Maximum tokens to generate
        limit: Optional limit on number of samples
        progress_interval: Print progress every N samples
    """
    print(f"Running inference on {len(records)} samples...")
    print(f"  Output: {output_file}")
    print(f"  Max new tokens: {max_new_tokens}")
    if limit:
        print(f"  Limit: {limit} samples")
    print()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    correct = 0
    evaluated = 0
    start_time = datetime.now()
    
    # Disable tqdm for cleaner output with running accuracy
    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, record in enumerate(records, 1):
            # Check limit
            if limit and idx > limit:
                break
            
            # Extract messages and image path
            messages = record.get("messages", [])
            image_path = record.get("image_path")
            if not image_path:
                images = record.get("images", [])
                if images:
                    image_path = images[0]
            
            if not messages or not image_path:
                failed += 1
                continue
            
            # Load image
            # Handle both relative paths (images/000000.png) and direct paths (000000.png)
            if image_path.startswith("images/"):
                # Strip "images/" prefix since media_dir already points to images directory
                image_filename = image_path.replace("images/", "", 1)
                full_image_path = os.path.join(media_dir, image_filename)
            else:
                full_image_path = os.path.join(media_dir, image_path)
            
            if not os.path.exists(full_image_path):
                # Debug: print first few failures
                if failed < 3:
                    print(f"Warning: Image not found: {full_image_path}")
                    print(f"  Media dir: {media_dir}")
                    print(f"  Image path from record: {image_path}")
                failed += 1
                continue
            
            try:
                image = Image.open(full_image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load image {full_image_path}: {e}")
                failed += 1
                continue
            
            # Prepare inputs
            try:
                # Try to apply chat template if processor supports it
                if hasattr(processor, 'apply_chat_template'):
                    # Use chat template for conversation format
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = processor(
                        text=text,
                        images=image,
                        return_tensors="pt"
                    ).to(device)
                else:
                    # Fallback: construct prompt manually from messages
                    prompt_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "system":
                            prompt_parts.append(f"System: {content}")
                        elif role == "user":
                            prompt_parts.append(f"User: {content}")
                    text = "\n".join(prompt_parts)
                    
                    inputs = processor(
                        text=text,
                        images=image,
                        return_tensors="pt"
                    ).to(device)
            except Exception as e:
                print(f"Warning: Failed to process inputs for sample {idx}: {e}")
                failed += 1
                continue
            
            # Generate prediction with no gradients
            with torch.no_grad():
                try:
                    # Get input length to extract only generated tokens
                    input_length = inputs.input_ids.shape[1]
                    
                    # Generate with deterministic decoding
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy decoding
                    )
                    
                    # Extract only the generated tokens
                    generated_ids_trimmed = generated_ids[0][input_length:]
                    
                    # Decode
                    output_text = processor.decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ).strip()
                    
                    # Prepare result
                    result = {
                        "sample_id": record.get("sample_id"),
                        "instruction": record.get("instruction"),
                        "image_path": image_path,
                        "output": output_text,
                    }
                    
                    # Include ground truth bbox if available
                    bbox = record.get("bbox") or record.get("gt_bbox")
                    if bbox:
                        result["bbox"] = bbox
                        result["gt_bbox"] = bbox
                    
                    # Include other metadata
                    for key in ["id", "application", "platform", "img_size", "ui_type", "group"]:
                        if key in record:
                            result[key] = record[key]
                    
                    # Evaluate accuracy if bbox available
                    is_correct = False
                    if bbox:
                        coords = parse_coordinates(output_text)
                        if coords:
                            x, y = coords
                            is_correct = point_in_bbox(x, y, bbox)
                            if is_correct:
                                correct += 1
                            evaluated += 1
                    
                    # Write result
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    successful += 1
                    
                except Exception as e:
                    print(f"Warning: Generation failed for sample {idx}: {e}")
                    failed += 1
                    continue
            
            # Progress update with running accuracy
            if idx % progress_interval == 0 or idx == 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = idx / elapsed if elapsed > 0 else 0
                accuracy = (correct / evaluated * 100) if evaluated > 0 else 0
                total_to_process = limit if limit else len(records)
                
                print(f"  [{idx}/{total_to_process}] "
                      f"Success: {successful}, Failed: {failed} | "
                      f"Accuracy: {correct}/{evaluated} ({accuracy:.2f}%) | "
                      f"Rate: {rate:.2f} samples/s")
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    total_processed = successful + failed
    final_accuracy = (correct / evaluated * 100) if evaluated > 0 else 0
    
    print()
    print("=" * 80)
    print("Inference complete!")
    print("=" * 80)
    print(f"  Total samples: {total_processed}")
    print(f"  ✓ Successful: {successful} ({successful/total_processed*100:.1f}%)")
    print(f"  ✗ Failed: {failed} ({failed/total_processed*100:.1f}%)")
    print()
    print(f"  🎯 Accuracy: {correct}/{evaluated} ({final_accuracy:.2f}%)")
    print(f"  ⏱  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  📊 Rate: {total_processed/elapsed:.2f} samples/s")
    print(f"  💾 Predictions: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run ScreenSpot-Pro benchmark using HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default model
  python scripts/run_screenspot_benchmark.py \\
    --model-name-or-path Asanshay/websight-v2-grounded \\
    --data-dir screenspot_pro
  
  # Run with LoRA adapter
  python scripts/run_screenspot_benchmark.py \\
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \\
    --adapter-path ckpts/checkpoint-200 \\
    --data-dir screenspot_pro
  
  # Run on subset for testing
  python scripts/run_screenspot_benchmark.py \\
    --model-name-or-path Asanshay/websight-v2-grounded \\
    --data-dir screenspot_pro \\
    --limit 100
        """
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Asanshay/websight-v2-grounded",
        help="HuggingFace model ID or local path (default: Asanshay/websight-v2-grounded)"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional path to LoRA adapter"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="screenspot_pro",
        help="Directory containing prepared ScreenSpot-Pro dataset (data.jsonl) (default: screenspot_pro)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory containing images (default: data-dir/images). Use if images are stored separately."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions (default: runs/screenspot_pro/<timestamp>.jsonl)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for testing (default: None)"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N samples (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"runs/screenspot_pro/predictions_{timestamp}.jsonl"
    
    print("=" * 80)
    print("ScreenSpot-Pro Benchmark")
    print("=" * 80)
    print(f"Model: {args.model_name_or_path}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Load dataset
    try:
        records, media_dir = load_screenspot_pro(args.data_dir, args.images_dir)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print()
        print("Please prepare the dataset first:")
        if args.images_dir:
            print(f"  python scripts/prepare_screenspot_pro.py --output-dir {args.data_dir} --images-dir {args.images_dir}")
        else:
            print(f"  python scripts/prepare_screenspot_pro.py --output-dir {args.data_dir}")
        return 1
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return 1
    
    if not records:
        print("✗ Error: No records found in dataset")
        return 1
    
    # Load model
    try:
        model, processor = load_model(
            args.model_name_or_path,
            args.adapter_path,
            args.device
        )
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Run inference
    try:
        run_inference(
            model,
            processor,
            records,
            media_dir,
            args.device,
            args.output,
            args.max_new_tokens,
            args.limit,
            args.progress_interval
        )
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("Next steps:")
    print("  Evaluate predictions:")
    print(f"    python scripts/evaluate_screenspot.py \\")
    print(f"      --predictions {args.output} \\")
    print(f"      --ground-truth {args.data_dir}/data.jsonl")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

