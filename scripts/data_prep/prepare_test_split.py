#!/usr/bin/env python3
"""
Extract and transform test split from wave-ui dataset for inference evaluation.

This script prepares the test dataset for inference-only evaluation (no gradients):
1. Reads records from wave-ui/data.jsonl or base/prompts/prompts.jsonl
2. Filters for test split
3. Transforms to LLaMA-Factory format (ShareGPT format, same as training)
4. Saves to data/wave_ui_test.jsonl

The output format matches training input format exactly:
- ShareGPT format with "messages" and "images" fields
- System prompt included (same as training)
- User message contains "<image>\n{prompt}" (same format as training)
- Processor will automatically apply qwen2_vl template during inference
- No ground truth/assistant messages needed for inference
"""

import json
import random
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

# System prompt used in training (must match exactly)
SYSTEM_PROMPT = "You are a GUI automation assistant. Given an image and a user instruction, output the exact pyautogui.click(x, y) command to execute the action. Coordinates are normalized to 1400x800 resolution."


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get the center point of a bounding box.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max] in original resolution
    
    Returns:
        Tuple of (center_x, center_y)
    """
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return center_x, center_y

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


def transform_record_from_prompts(record: Dict, base_image_path: str = "wave-ui") -> Dict:
    """
    Transform a record from prompts.jsonl format to LLaMA-Factory format for inference.
    
    Includes system prompt and user message (same format as training, no ground truth needed for inference).

    Args:
        record: Record from prompts.jsonl with 'original' and 'prompt' fields
        base_image_path: Base path for images

    Returns:
        Transformed record with system prompt, user message, and image (for inference)
    """
    original = record["original"]
    prompt = record["prompt"]

    # Extract image path
    raw_image_path = original["image_path"]
    if os.path.isabs(raw_image_path):
        relative_image_path = os.path.relpath(raw_image_path, base_image_path)
    else:
        relative_image_path = raw_image_path
    relative_image_path = relative_image_path.replace("\\", "/")

    # Format for LLaMA-Factory (ShareGPT-style) with <image> placeholder
    # This format matches training input format exactly - includes system prompt
    # Processor will apply qwen2_vl template automatically
    transformed = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},  # Same system prompt as training
            {"role": "user", "content": f"<image>\n{prompt}"},  # Same format as training input
        ],
        "images": [
            relative_image_path
        ],
    }

    return transformed


def transform_record_from_waveui(record: Dict, base_image_path: str = "wave-ui") -> Dict:
    """
    Transform a record from wave-ui/data.jsonl format to LLaMA-Factory format for inference.
    Uses the 'instruction' field as the prompt.
    
    Includes system prompt and user message (same format as training, no ground truth needed for inference).

    Args:
        record: Record from wave-ui/data.jsonl
        base_image_path: Base path for images

    Returns:
        Transformed record with system prompt, user message, and image (for inference)
    """
    # Extract data
    raw_image_path = record["image_path"]
    instruction = record.get("instruction", "")
    
    if os.path.isabs(raw_image_path):
        relative_image_path = os.path.relpath(raw_image_path, base_image_path)
    else:
        relative_image_path = raw_image_path
    relative_image_path = relative_image_path.replace("\\", "/")

    # Format for LLaMA-Factory (ShareGPT-style) with <image> placeholder
    # This format matches training input format exactly - includes system prompt
    # Processor will apply qwen2_vl template automatically
    transformed = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},  # Same system prompt as training
            {"role": "user", "content": f"<image>\n{instruction}"},  # Same format as training input
        ],
        "images": [
            relative_image_path
        ],
    }

    return transformed


def main():
    parser = argparse.ArgumentParser(
        description="Extract and transform test split for inference evaluation (no gradients)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script prepares the test dataset for inference-only evaluation.
The output format matches training input format exactly (ShareGPT format with qwen2_vl template).
Includes system prompt and user messages (same as training, no ground truth needed for inference).

The test file will be used by test_after_grounding.py for inference evaluation.
Predictions will be saved to a separate JSONL file during inference.
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file path (wave-ui/data.jsonl or base/prompts/prompts.jsonl). If not specified, will try prompts.jsonl first, then wave-ui/data.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wave_ui_test.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--base-image-path",
        type=str,
        default="wave-ui",
        help="Base path for image files in output",
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Auto-detect input file if not specified
    if args.input is None:
        prompts_path = project_root / "base" / "prompts" / "prompts.jsonl"
        waveui_path = project_root / "wave-ui" / "data.jsonl"
        
        # Prefer prompts.jsonl if it exists
        if prompts_path.exists():
            input_path = prompts_path
            use_prompts = True
            print(f"Auto-detected input: {input_path} (using prompts.jsonl format)")
        elif waveui_path.exists():
            input_path = waveui_path
            use_prompts = False
            print(f"Auto-detected input: {input_path} (using wave-ui/data.jsonl format)")
        else:
            print(f"Error: Could not find input file. Tried:")
            print(f"  - {prompts_path}")
            print(f"  - {waveui_path}")
            return 1
    else:
        input_path = Path(args.input)
        # Detect format based on file structure
        # Check first line to determine format
        with open(input_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line.strip():
                first_record = json.loads(first_line)
                use_prompts = "original" in first_record and "prompt" in first_record
            else:
                use_prompts = False
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Reading dataset from {input_path}...")

    # Read all records and filter for test split
    test_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Check if this is a test split record
                if use_prompts:
                    # For prompts.jsonl format
                    if record.get("original", {}).get("split") == "test":
                        test_records.append(record)
                else:
                    # For wave-ui/data.jsonl format
                    if record.get("split") == "test":
                        test_records.append(record)

    print(f"Found {len(test_records)} test records")

    if len(test_records) == 0:
        print("Error: No test records found!")
        return 1

    # Transform and save test set
    print(f"Transforming and saving test set to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in test_records:
            if use_prompts:
                transformed = transform_record_from_prompts(record, args.base_image_path)
            else:
                transformed = transform_record_from_waveui(record, args.base_image_path)
            f.write(json.dumps(transformed, ensure_ascii=False) + "\n")

    print(f"Saved {len(test_records)} test records to {output_path}")

    # Print sample
    if len(test_records) > 0:
        print("\nSample transformed record:")
        if use_prompts:
            sample = transform_record_from_prompts(test_records[0], args.base_image_path)
        else:
            sample = transform_record_from_waveui(test_records[0], args.base_image_path)
        print(json.dumps(sample, indent=2, ensure_ascii=False))

    print("\n✓ Test split preparation complete!")
    return 0


if __name__ == "__main__":
    exit(main())

