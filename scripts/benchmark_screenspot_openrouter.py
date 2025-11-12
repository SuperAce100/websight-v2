#!/usr/bin/env python3
"""
Benchmark model on ScreenSpot-Pro dataset using OpenRouter API.
Downloads from HuggingFace, transforms to ShareGPT format, and runs inference via OpenRouter.

Usage:
python scripts/benchmark_screenspot_openrouter.py --api-key "API-KEY-HERE" --model "qwen/qwen3-vl-8b-instruct"
"""

import json
import argparse
import os
import base64
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI
from PIL import Image as PILImage
from io import BytesIO


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def download_screenspot_pro(output_dir: str = "screenspot_pro", download_delay: float = 0.5):
    """Download ScreenSpot-Pro dataset from HuggingFace and transform to proper format.
    
    The dataset structure on HuggingFace:
    - annotations/ folder with JSON files containing annotation entries
    - images/ folder with subdirectories (e.g., blender_windows/) containing images
    
    Each annotation entry contains:
    - img_filename: path to image (e.g., "blender_windows/screenshot_2024-12-02_13-33-12.png")
    - bbox: bounding box coordinates [x1, y1, x2, y2]
    - instruction: task instruction text
    - Other metadata: id, application, platform, img_size, ui_type, group, etc.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface-hub")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Download dataset from HuggingFace
    raw_dir = output_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ScreenSpot-Pro dataset from HuggingFace...")
    print("(Will use cache if already downloaded)")
    
    # Retry logic for rate limiting with exponential backoff
    max_retries = 5
    retry_delay = 60  # Start with 60 seconds
    
    snapshot_path = None
    for attempt in range(max_retries):
        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id="likaixin/ScreenSpot-Pro",
                    repo_type="dataset",
                    local_dir=raw_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=("annotations/*.json", "images/**/*.png", "images/**/*.jpg", "README.md", "LICENSE.md"),
                )
            ).expanduser().resolve()
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower() or "Too Many Requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 60s, 120s, 240s, 480s
                    print(f"\n⚠️  Rate limited. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                    print("   Tip: Make sure you're logged in with 'huggingface-cli login'")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {max_retries} attempts. Please wait a few minutes or ensure you're logged in with 'huggingface-cli login'")
            else:
                # Different error, re-raise
                raise
    
    if snapshot_path is None:
        raise Exception("Failed to download dataset after retries")
    
    annotations_dir = snapshot_path / "annotations"
    raw_images_dir = snapshot_path / "images"
    
    if not annotations_dir.exists():
        raise Exception(f"Annotations directory not found: {annotations_dir}")
    if not raw_images_dir.exists():
        raise Exception(f"Images directory not found: {raw_images_dir}")
    
    # Find all JSON annotation files
    annotation_files = list(annotations_dir.glob("*.json"))
    if not annotation_files:
        raise Exception(f"No JSON annotation files found in {annotations_dir}")
    
    print(f"Found {len(annotation_files)} annotation file(s)")
    
    # Load all annotations
    all_annotations = []
    for ann_file in annotation_files:
        print(f"Loading annotations from {ann_file.name}...")
        with open(ann_file, "r", encoding="utf-8") as f:
            try:
                # Try loading as JSON array
                data = json.load(f)
                if isinstance(data, list):
                    all_annotations.extend(data)
                elif isinstance(data, dict):
                    # If it's a dict, check for common keys that might contain the list
                    for key in ["annotations", "data", "samples", "items"]:
                        if key in data and isinstance(data[key], list):
                            all_annotations.extend(data[key])
                            break
                    else:
                        # If no list found, treat the dict itself as a single annotation
                        all_annotations.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {ann_file.name}: {e}")
                continue
    
    print(f"Loaded {len(all_annotations)} annotation entries")
    
    # System prompt for API calls
    system_prompt = "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."
    
    # Open JSONL file for incremental writing (so progress isn't lost if interrupted)
    jsonl_path = output_path / "data.jsonl"
    records = []
    
    print("Processing annotations and images...")
    print(f"Results will be saved incrementally to: {jsonl_path}")
    
    # Write incrementally so progress isn't lost if interrupted
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, ann in enumerate(tqdm(all_annotations, desc="Processing samples")):
            # Extract image filename from annotation
            img_filename = ann.get("img_filename")
            if not img_filename:
                print(f"Warning: Sample {i} missing img_filename, skipping")
                continue
            
            # Construct full path to source image
            source_image_path = raw_images_dir / img_filename
            
            if not source_image_path.exists():
                print(f"Warning: Image not found: {source_image_path}, skipping")
                continue
            
            # Create a simple numbered filename for output
            # Preserve original extension
            img_ext = Path(img_filename).suffix or ".png"
            output_image_filename = f"{i:06d}{img_ext}"
            dest_image_path = images_dir / output_image_filename
            
            # Copy image to output directory
            try:
                shutil.copy2(source_image_path, dest_image_path)
            except Exception as e:
                print(f"Warning: Failed to copy image {source_image_path}: {e}, skipping")
                continue
            
            # Extract instruction
            instruction = ann.get("instruction", "")
            if not instruction:
                instruction = "click on the target element"
            
            # Extract bounding box
            bbox = ann.get("bbox")
            if bbox and isinstance(bbox, list) and len(bbox) >= 4:
                # Ensure bbox is in correct format [x1, y1, x2, y2]
                bbox = [int(coord) for coord in bbox[:4]]
            else:
                bbox = None
            
            # Create user text prompt
            user_text = f"Query: {instruction}\nOutput only the coordinate of one point in your response as pyautogui commands.\nFormat: pyautogui.click(x, y)\n"
            
            # Create record matching wave-ui format with all metadata
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"<image>\n{user_text}"  # Will be converted to proper format during API call
                    }
                ],
                "image_path": f"images/{output_image_filename}",  # Match wave-ui format
                "images": [f"images/{output_image_filename}"],  # Keep for backward compatibility
                "instruction": instruction,
                "user_text": user_text,  # Store the formatted text separately
                "sample_id": i,
            }
            
            # Add bounding box if available
            if bbox:
                record["bbox"] = bbox
                record["gt_bbox"] = bbox  # Also store as gt_bbox for evaluation
            
            # Preserve all other metadata fields from annotation
            for key, value in ann.items():
                if key not in ["img_filename", "bbox"]:  # Already processed
                    # Skip None values
                    if value is not None:
                        record[key] = value
            
            # Write immediately to preserve progress
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # Ensure data is written to disk
            records.append(record)
    
    print(f"Transformed {len(records)} samples to {jsonl_path}")
    return records, str(output_path)


def run_inference_openrouter(
    api_key: str,
    model: str,
    records: List[Dict],
    media_dir: str,
    output_file: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    rate_limit_delay: float = 0.05,
    progress_interval: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    limit: Optional[int] = None
):
    """Run inference on records using OpenRouter API and save predictions."""
    total_samples = len(records)
    print(f"Running inference on {total_samples} samples using OpenRouter API...")
    print(f"Model: {model}")
    print(f"Temperature: {temperature} | Max tokens: {max_tokens}")
    print(f"Progress updates every {progress_interval} samples")
    print(f"Results will be saved incrementally to: {output_file}\n")
    
    # Initialize OpenAI client for OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, record in enumerate(tqdm(records, desc="Inference"), 1):
            if limit is not None and idx > limit:
                break
            messages = record.get("messages", [])
            
            # Prefer image_path (wave-ui format), fall back to images array
            image_path = record.get("image_path")
            if not image_path:
                images = record.get("images", [])
                if images:
                    image_path = images[0]
            
            if not messages or not image_path:
                failed += 1
                continue
            
            full_image_path = os.path.join(media_dir, image_path)
            
            if not os.path.exists(full_image_path):
                failed += 1
                continue
            
            # Encode image to base64
            try:
                image_base64 = encode_image_to_base64(full_image_path)
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")
                failed += 1
                continue
            
            # Prepare messages for OpenRouter API with improved prompt structure
            api_messages = []
            # Determine image size to instruct model to return ORIGINAL pixel coordinates
            img_width: Optional[int] = None
            img_height: Optional[int] = None
            try:
                with PILImage.open(full_image_path) as im_sz:
                    img_width, img_height = im_sz.size
            except Exception:
                pass
            for msg in messages:
                if msg["role"] == "system":
                    # System message - use structured format
                    api_messages.append({
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": msg["content"]
                            }
                        ]
                    })
                elif msg["role"] == "user" and "<image>" in msg["content"]:
                    # User message with image - use improved format
                    # Build prompt using instruction and explicit ORIGINAL pixel requirement
                    instruction = record.get("instruction")
                    if isinstance(instruction, str) and instruction.strip():
                        size_line = f"Image size: {img_width}x{img_height} (width x height). " if (img_width and img_height) else ""
                        text_content = (
                            f"Query: {instruction}\n"
                            f"{size_line}Return coordinates in ORIGINAL pixels of the image shown.\n"
                            f"Output only the coordinate of one point in your response as pyautogui commands.\n"
                            f"Format: pyautogui.click(x, y)\n"
                        )
                    else:
                        # Fallback to provided user_text or raw content
                        if "user_text" in record:
                            text_content = record["user_text"]
                        else:
                            text_content = msg["content"].replace("<image>", "").strip()
                    
                    api_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": text_content
                            }
                        ]
                    })
                else:
                    # Regular text message
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Make API request with retries
            output_text = None
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        extra_body={},
                        model=model,
                        messages=api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    output_text = completion.choices[0].message.content
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    # Check for rate limiting
                    if "429" in error_str or "rate limit" in error_str.lower():
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"API error: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (2 ** attempt))
                            continue
                        else:
                            break
                
                # Small delay between requests to avoid rate limiting
                time.sleep(rate_limit_delay)
            
            if output_text is None:
                failed += 1
            else:
                # Save result
                result = {
                    "input": {
                        "prompt": messages[1]["content"] if len(messages) > 1 else "",
                        "image_path": image_path,
                    },
                    "output": output_text,
                    "sample_id": record.get("sample_id"),
                    "instruction": record.get("instruction"),
                }
                
                # Include bounding box information
                if "bbox" in record:
                    result["bbox"] = record["bbox"]
                if "gt_bbox" in record:
                    result["gt_bbox"] = record["gt_bbox"]
                if "gt_label" in record:
                    result["gt_label"] = record.get("gt_label")
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()  # Ensure data is written to disk immediately
                successful += 1
            
            # Periodic progress update (for all samples, success or failure)
            if idx % progress_interval == 0 or idx == total_samples:
                elapsed_time = time.time() - start_time
                processed = successful + failed
                percentage = (processed / total_samples) * 100
                avg_time_per_sample = elapsed_time / processed if processed > 0 else 0
                remaining_samples = total_samples - processed
                estimated_remaining_time = avg_time_per_sample * remaining_samples
                
                print(f"\n[Progress Update] Sample {idx}/{total_samples} ({percentage:.1f}%)")
                print(f"  ✓ Successful: {successful} | ✗ Failed: {failed} | Total processed: {processed}")
                print(f"  ⏱️  Elapsed: {elapsed_time:.1f}s | Avg: {avg_time_per_sample:.2f}s/sample")
                if remaining_samples > 0:
                    print(f"  ⏳ Estimated remaining: {estimated_remaining_time:.1f}s ({estimated_remaining_time/60:.1f} min)")
                print(f"  💾 Results saved to: {output_path}\n")
            
            # Small delay between requests to avoid rate limiting
            time.sleep(rate_limit_delay)
    
    total_time = time.time() - start_time
    print(f"\n✓ Inference complete!")
    print(f"  Total samples: {total_samples}")
    print(f"  ✓ Successful: {successful} ({successful/total_samples*100:.1f}%)")
    print(f"  ✗ Failed: {failed} ({failed/total_samples*100:.1f}%)")
    print(f"  ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  📊 Average: {total_time/total_samples:.2f}s per sample")
    print(f"  💾 Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model on ScreenSpot-Pro dataset using OpenRouter API")
    parser.add_argument("--api-key", type=str, required=True, help="OpenRouter API key")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'qwen/qwen3-vl-8b-instruct', 'openai/gpt-4o', 'google/gemini-pro-vision', 'anthropic/claude-3.5-sonnet')")
    parser.add_argument("--output-dir", type=str, default="screenspot_pro", help="Output directory for dataset")
    parser.add_argument("--predictions", type=str, default="screenspot_pro_predictions_openrouter.jsonl", help="Output predictions file")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if dataset already exists")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for API calls")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Initial delay between retries (seconds)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.05, help="Delay between API requests (seconds, default: 0.05 for faster processing)")
    parser.add_argument("--progress-interval", type=int, default=10, help="Print progress update every N samples (default: 10)")
    parser.add_argument("--download-delay", type=float, default=0.5, help="(Deprecated: not used) Delay parameter kept for compatibility. Processing is fast and doesn't need delays.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate (default: 2048)")
    parser.add_argument("--limit", type=int, help="Limit number of samples for inference (debugging)")
    
    args = parser.parse_args()
    
    # Download and transform dataset
    data_jsonl = Path(args.output_dir) / "data.jsonl"
    if args.skip_download and data_jsonl.exists():
        print(f"Loading existing dataset from {data_jsonl}...")
        records = []
        with open(data_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        media_dir = args.output_dir
    else:
        records, media_dir = download_screenspot_pro(args.output_dir, download_delay=args.download_delay)
    
    if not records:
        print("Error: No records found!")
        return 1
    
    # Run inference via OpenRouter
    run_inference_openrouter(
        api_key=args.api_key,
        model=args.model,
        records=records,
        media_dir=media_dir,
        output_file=args.predictions,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        rate_limit_delay=args.rate_limit_delay,
        progress_interval=args.progress_interval,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit
    )
    
    print(f"\n✓ Benchmark complete! Predictions saved to: {args.predictions}")
    return 0


if __name__ == "__main__":
    exit(main())

