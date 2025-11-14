#!/usr/bin/env python3
"""
Shared utilities for ScreenSpot-Pro dataset preparation.

This module provides functions to download and transform the ScreenSpot-Pro dataset
from HuggingFace, preserving all metadata including bounding boxes for evaluation.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def download_screenspot_pro(
    output_dir: str = "screenspot_pro",
    max_retries: int = 5,
    retry_delay: int = 60
) -> Tuple[List[Dict], str]:
    """
    Download ScreenSpot-Pro dataset from HuggingFace and transform to proper format.
    
    The dataset structure on HuggingFace:
    - annotations/ folder with JSON files containing annotation entries
    - images/ folder with subdirectories (e.g., blender_windows/) containing images
    
    Each annotation entry contains:
    - img_filename: path to image (e.g., "blender_windows/screenshot_2024-12-02_13-33-12.png")
    - bbox: bounding box coordinates [x1, y1, x2, y2]
    - instruction: task instruction text
    - Other metadata: id, application, platform, img_size, ui_type, group, etc.
    
    Args:
        output_dir: Directory to save processed dataset
        max_retries: Maximum number of retry attempts for rate limiting
        retry_delay: Initial delay in seconds between retries (exponential backoff)
    
    Returns:
        Tuple of (records list, media directory path)
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "Please install huggingface_hub: pip install huggingface-hub"
        )
    
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
    snapshot_path = None
    for attempt in range(max_retries):
        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id="likaixin/ScreenSpot-Pro",
                    repo_type="dataset",
                    local_dir=raw_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=(
                        "annotations/*.json",
                        "images/**/*.png",
                        "images/**/*.jpg",
                        "README.md",
                        "LICENSE.md"
                    ),
                )
            ).expanduser().resolve()
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower() or "Too Many Requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(
                        f"\n⚠️  Rate limited. Waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    print("   Tip: Make sure you're logged in with 'huggingface-cli login'")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(
                        f"Rate limit exceeded after {max_retries} attempts. "
                        "Please wait a few minutes or ensure you're logged in with "
                        "'huggingface-cli login'"
                    )
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
    
    # System prompt for ScreenSpot-Pro (expert GUI interaction)
    system_prompt = (
        "You are an expert in using electronic devices and interacting with graphic "
        "interfaces. You should not call any external tools."
    )
    
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
            user_text = (
                f"Query: {instruction}\n"
                "Output only the coordinate of one point in your response as pyautogui commands.\n"
                "Format: pyautogui.click(x, y)\n"
            )
            
            # Create record matching ShareGPT format with all metadata
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"<image>\n{user_text}"
                    }
                ],
                "image_path": f"images/{output_image_filename}",
                "images": [f"images/{output_image_filename}"],
                "instruction": instruction,
                "user_text": user_text,
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


def load_screenspot_pro(data_dir: str = "screenspot_pro") -> Tuple[List[Dict], str]:
    """
    Load existing ScreenSpot-Pro dataset from disk.
    
    Args:
        data_dir: Directory containing the processed dataset
    
    Returns:
        Tuple of (records list, media directory path)
    """
    data_path = Path(data_dir)
    jsonl_path = data_path / "data.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {jsonl_path}. "
            "Please run download_screenspot_pro() first."
        )
    
    print(f"Loading existing dataset from {jsonl_path}...")
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records")
    return records, str(data_dir)

