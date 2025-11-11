#!/usr/bin/env python3
"""
Transform wave-ui/prompts.jsonl dataset for Qwen3-VL fine-tuning.

This script:
1. Reads each record from the original dataset
2. Samples a random click location within the bounding box
3. Normalizes coordinates from original resolution to 1400x800
4. Outputs in LLaMA-Factory conversation format with PyAutoGUI commands
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def sample_click_location(bbox: List[float], seed: int = None) -> Tuple[float, float]:
    """
    Sample a random click location within the bounding box.

    Args:
        bbox: [x_min, y_min, x_max, y_max] in original resolution
        seed: Optional seed for reproducibility

    Returns:
        Tuple of (x, y) coordinates sampled within the bbox
    """
    if seed is not None:
        random.seed(seed)

    x_min, y_min, x_max, y_max = bbox
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)

    return x, y


def normalize_coordinates(
    x: float,
    y: float,
    original_width: int,
    original_height: int,
    target_width: int = 1400,
    target_height: int = 800,
) -> Tuple[int, int]:
    """
    Normalize coordinates from original resolution to target resolution.

    Args:
        x, y: Coordinates in original resolution
        original_width, original_height: Original image dimensions
        target_width, target_height: Target normalized dimensions

    Returns:
        Tuple of normalized (x, y) coordinates rounded to integers
    """
    norm_x = (x / original_width) * target_width
    norm_y = (y / original_height) * target_height

    return round(norm_x), round(norm_y)


def transform_record(record: Dict, base_image_path: str = "wave-ui") -> Dict:
    """
    Transform a single record to LLaMA-Factory format.

    Args:
        record: Original dataset record
        base_image_path: Base path for images

    Returns:
        Transformed record in conversation format
    """
    original = record["original"]

    # Extract data
    bbox = original["bbox"]
    resolution = original["resolution"]  # [width, height]
    image_path = original["image_path"]
    prompt = record["prompt"]

    # Sample click location within bbox
    click_x, click_y = sample_click_location(bbox, seed=record.get("id"))

    # Normalize to 1400x800
    norm_x, norm_y = normalize_coordinates(
        click_x, click_y, resolution[0], resolution[1]
    )

    # Format for LLaMA-Factory with system prompt
    # Use relative path from images folder (not full absolute path)
    # The image_path from the dataset already includes "images/" prefix
    transformed = {
        "messages": [
            {
                "role": "system",
                "content": "You are a GUI automation assistant. Given an image and a user instruction, output the exact pyautogui.click(x, y) command to execute the action. Coordinates are normalized to 1400x800 resolution.",
            },
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": f"pyautogui.click({norm_x}, {norm_y})"},
        ],
        "images": [
            image_path
        ],  # Store relative path, --image_folder will provide the base
    }

    return transformed


def split_dataset(
    records: List[Dict], val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.

    Args:
        records: List of all records
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_records, val_records)
    """
    random.seed(seed)

    # Check if records already have a 'split' field
    train_records = [
        r for r in records if r.get("original", {}).get("split") == "train"
    ]
    test_records = [r for r in records if r.get("original", {}).get("split") == "test"]

    if train_records or test_records:
        print(
            f"Using existing split: {len(train_records)} train, {len(test_records)} test"
        )
        return train_records, test_records

    # Otherwise, create random split
    shuffled = records.copy()
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

    print(
        f"Created random split: {len(train_records)} train, {len(val_records)} validation"
    )
    return train_records, val_records


def main():
    parser = argparse.ArgumentParser(
        description="Transform wave-ui dataset for Qwen3-VL training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/hai/scratch/asanshay/websight-v2/data/prompts.jsonl",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for transformed files",
    )
    parser.add_argument(
        "--base-image-path",
        type=str,
        default="/hai/scratch/asanshay/websight-v2/data",
        help="Base path for image files in output",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (if no existing split)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset from {input_path}...")

    # Read all records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Split into train/val
    train_records, val_records = split_dataset(records, args.val_ratio, args.seed)

    # Transform and save train set
    train_output = output_dir / "wave_ui_train.jsonl"
    print(f"Transforming and saving training set to {train_output}...")

    with open(train_output, "w", encoding="utf-8") as f:
        for record in train_records:
            transformed = transform_record(record, args.base_image_path)
            f.write(json.dumps(transformed, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_records)} training records")

    # Transform and save validation set
    val_output = output_dir / "wave_ui_val.jsonl"
    print(f"Transforming and saving validation set to {val_output}...")

    with open(val_output, "w", encoding="utf-8") as f:
        for record in val_records:
            transformed = transform_record(record, args.base_image_path)
            f.write(json.dumps(transformed, ensure_ascii=False) + "\n")

    print(f"Saved {len(val_records)} validation records")

    # Print sample
    print("\nSample transformed record:")
    print(
        json.dumps(
            transform_record(records[0], args.base_image_path),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n✓ Transformation complete!")


if __name__ == "__main__":
    main()
