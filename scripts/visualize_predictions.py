#!/usr/bin/env python3
"""Visualize predictions and bounding boxes on images."""

import json
import re
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional


def extract_coordinates(output_text: str) -> Optional[Tuple[int, int]]:
    """Extract (x, y) coordinates from model output.
    
    Handles formats like:
    - "pyautogui.click(572, 252)"
    - "343, 312"
    - "x=572, y=252"
    """
    # Try pyautogui.click(x, y) format
    match = re.search(r'pyautogui\.click\((\d+),\s*(\d+)\)', output_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    # Try simple "x, y" format
    match = re.search(r'(\d+),\s*(\d+)', output_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    # Try x=..., y=... format
    x_match = re.search(r'x\s*=\s*(\d+)', output_text, re.IGNORECASE)
    y_match = re.search(r'y\s*=\s*(\d+)', output_text, re.IGNORECASE)
    if x_match and y_match:
        return (int(x_match.group(1)), int(y_match.group(1)))
    
    return None


def draw_bbox_and_prediction(
    image: Image.Image,
    bbox: list,
    pred_point: Optional[Tuple[int, int]],
    instruction: str,
    sample_id: int
) -> Image.Image:
    """Draw bounding box and prediction point on image."""
    # Create a copy to draw on
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        except:
            font = None
            small_font = None
    
    # Draw bounding box (green)
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        # Draw rectangle outline
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        # Draw center point of bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        draw.ellipse([center_x - 5, center_y - 5, center_x + 5, center_y + 5], 
                    fill="green", outline="darkgreen", width=2)
        # Label for bbox center
        if font:
            draw.text((x1, y1 - 25), "GT Center", fill="green", font=small_font)
    
    # Draw prediction point (red)
    if pred_point:
        px, py = pred_point
        # Draw a larger circle for visibility
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], 
                    fill="red", outline="darkred", width=2)
        # Draw crosshair
        draw.line([px - 15, py, px + 15, py], fill="red", width=2)
        draw.line([px, py - 15, px, py + 15], fill="red", width=2)
        # Label for prediction
        if font:
            draw.text((px + 12, py - 25), f"Pred: ({px}, {py})", fill="red", font=small_font)
    
    # Add instruction text at the top
    if font:
        # Create a semi-transparent background for text
        text_bg = Image.new('RGBA', img.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_bg)
        text_bbox = text_draw.textbbox((10, 10), f"ID: {sample_id} | {instruction}", font=font)
        text_draw.rectangle([text_bbox[0] - 5, text_bbox[1] - 5, 
                           text_bbox[2] + 5, text_bbox[3] + 5], 
                          fill=(0, 0, 0, 200))
        img = Image.alpha_composite(img.convert('RGBA'), text_bg).convert('RGB')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"ID: {sample_id} | {instruction}", fill="white", font=font)
    
    return img


def visualize_predictions(
    predictions_file: str,
    images_dir: str,
    output_dir: str,
    sample_ids: list = None
):
    """Visualize predictions and bounding boxes for specified samples."""
    predictions_path = Path(predictions_file)
    images_base = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read predictions
    records = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Filter by sample_ids if provided
    if sample_ids is not None:
        records = [r for r in records if r.get('sample_id') in sample_ids]
    
    print(f"Processing {len(records)} samples...")
    
    for record in records:
        sample_id = record.get('sample_id')
        image_path = record.get('input', {}).get('image_path', '')
        bbox = record.get('bbox') or record.get('gt_bbox')
        instruction = record.get('instruction', '')
        output_text = record.get('output', '')
        
        if not image_path:
            print(f"Warning: No image_path for sample {sample_id}")
            continue
        
        # Load image
        full_image_path = images_base / image_path
        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        try:
            image = Image.open(full_image_path)
        except Exception as e:
            print(f"Warning: Failed to load image {full_image_path}: {e}")
            continue
        
        # Extract prediction coordinates
        pred_point = extract_coordinates(output_text)
        if pred_point is None:
            print(f"Warning: Could not extract coordinates from: {output_text}")
        
        # Draw annotations
        annotated_img = draw_bbox_and_prediction(
            image, bbox, pred_point, instruction, sample_id
        )
        
        # Save annotated image
        output_filename = f"sample_{sample_id:06d}_annotated.png"
        output_file = output_path / output_filename
        annotated_img.save(output_file)
        print(f"Saved: {output_file}")
    
    print(f"\n✓ Visualization complete! Images saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize predictions and bounding boxes")
    parser.add_argument("--predictions", type=str, 
                       default="screenspot_pro_predictions_openrouter.jsonl",
                       help="Path to predictions JSONL file")
    parser.add_argument("--images-dir", type=str, default="screenspot_pro",
                       help="Base directory containing images")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: temp directory)")
    parser.add_argument("--sample-ids", type=int, nargs="+", default=[118, 119, 120, 121, 122, 123],
                       help="Sample IDs to visualize (default: 118-123)")
    
    args = parser.parse_args()
    
    # Use temp directory if not specified
    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="screenspot_viz_"))
        print(f"Using temporary directory: {output_dir}")
    else:
        output_dir = args.output_dir
    
    visualize_predictions(
        args.predictions,
        args.images_dir,
        output_dir,
        args.sample_ids
    )

