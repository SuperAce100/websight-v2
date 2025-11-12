#!/usr/bin/env python3
"""
Evaluation script for grounding protocol on Wave UI dataset.

Evaluates if predicted click coordinates fall within ground truth bounding boxes.
The model outputs pyautogui.click(x, y) commands with coordinates normalized to 1400x800.
Ground truth bboxes are in original resolution and need to be normalized for comparison.
"""

import json
import re
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def normalize_bbox(
    bbox: List[float],
    original_width: int,
    original_height: int,
    target_width: int = 1400,
    target_height: int = 800,
) -> List[float]:
    """
    Normalize bounding box coordinates from original resolution to target resolution.

    Args:
        bbox: [x_min, y_min, x_max, y_max] in original resolution
        original_width, original_height: Original image dimensions
        target_width, target_height: Target normalized dimensions (default: 1400x800)

    Returns:
        Normalized bbox [x_min, y_min, x_max, y_max] in target resolution
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Normalize coordinates
    norm_x_min = (x_min / original_width) * target_width
    norm_y_min = (y_min / original_height) * target_height
    norm_x_max = (x_max / original_width) * target_width
    norm_y_max = (y_max / original_height) * target_height
    
    return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]


def parse_pyautogui_click(text: str) -> Optional[Tuple[int, int]]:
    """
    Parse pyautogui.click(x, y) command from text.

    Args:
        text: Text that may contain pyautogui.click(x, y) command

    Returns:
        Tuple of (x, y) coordinates if found, None otherwise
    """
    # Pattern to match pyautogui.click(x, y) or pyautogui.click(x,y)
    pattern = r'pyautogui\.click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    match = re.search(pattern, text)
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x, y)
    
    return None


def point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
    """
    Check if a point is within a bounding box.

    Args:
        point: (x, y) coordinates
        bbox: [x_min, y_min, x_max, y_max]

    Returns:
        True if point is within bbox, False otherwise
    """
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    
    return x_min <= x <= x_max and y_min <= y <= y_max


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: [x_min, y_min, x_max, y_max]

    Returns:
        Tuple of (center_x, center_y) coordinates
    """
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return (center_x, center_y)


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: (x1, y1) coordinates
        point2: (x2, y2) coordinates

    Returns:
        Euclidean distance between the two points
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_image_path(image_path: str) -> str:
    """
    Normalize image path for consistent matching.
    Removes 'wave-ui/' prefix if present and strips leading/trailing slashes.
    
    Args:
        image_path: Image path string
        
    Returns:
        Normalized image path
    """
    if not image_path:
        return image_path
    
    # Remove wave-ui/ prefix if present
    if image_path.startswith("wave-ui/"):
        image_path = image_path[8:]
    elif image_path.startswith("./wave-ui/"):
        image_path = image_path[10:]
    
    # Strip leading/trailing slashes
    image_path = image_path.strip("/")
    
    return image_path


def load_ground_truth(prompts_path: Path) -> Dict[str, Dict]:
    """
    Load ground truth data from prompts.jsonl.

    Args:
        prompts_path: Path to prompts.jsonl file

    Returns:
        Dictionary mapping image path to ground truth data
    """
    ground_truth = {}
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                original = record.get("original", {})
                image_path = original.get("image_path")
                
                if image_path:
                    # Normalize image path for consistent matching
                    normalized_path = normalize_image_path(image_path)
                    
                    ground_truth[normalized_path] = {
                        "id": record.get("id"),
                        "bbox": original.get("bbox"),
                        "resolution": original.get("resolution"),
                        "prompt": record.get("prompt"),
                        "original": original,
                        "original_image_path": image_path  # Keep original for reference
                    }
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    return ground_truth


def load_predictions(predictions_path: Path) -> Dict[str, str]:
    """
    Load model predictions from a JSONL file.

    Expected format: Each line is a JSON object with:
    - "image_path" or "image" or "images": path to image
    - "prediction" or "output" or "response" or "messages": model output text

    Args:
        predictions_path: Path to predictions JSONL file

    Returns:
        Dictionary mapping image path to prediction text
    """
    predictions = {}
    
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                
                # Try different possible keys for image path
                image_path = None
                if "image_path" in record:
                    image_path = record["image_path"]
                elif "image" in record:
                    image_path = record["image"]
                elif "images" in record:
                    images = record["images"]
                    if isinstance(images, list) and len(images) > 0:
                        image_path = images[0]
                    elif isinstance(images, str):
                        image_path = images
                
                # Try different possible keys for prediction
                prediction = None
                if "prediction" in record:
                    prediction = record["prediction"]
                elif "output" in record:
                    prediction = record["output"]
                elif "response" in record:
                    prediction = record["response"]
                elif "generated_text" in record:
                    prediction = record["generated_text"]
                elif "messages" in record:
                    messages = record["messages"]
                    if isinstance(messages, list) and len(messages) > 0:
                        # Get the last message (assistant response)
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            prediction = last_msg.get("content")
                
                if image_path and prediction:
                    # Normalize image path for consistent matching
                    normalized_path = normalize_image_path(image_path)
                    predictions[normalized_path] = prediction
                else:
                    if not image_path:
                        print(f"Warning: Line {line_num} missing image_path")
                    if not prediction:
                        print(f"Warning: Line {line_num} missing prediction")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    return predictions


def evaluate_grounding(
    ground_truth: Dict[str, Dict],
    predictions: Dict[str, str],
    target_width: int = 1400,
    target_height: int = 800,
) -> Dict:
    """
    Evaluate grounding by checking if predicted clicks are within ground truth bboxes.

    Args:
        ground_truth: Dictionary mapping image path to ground truth data
        predictions: Dictionary mapping image path to prediction text
        target_width: Target normalized width (default: 1400)
        target_height: Target normalized height (default: 800)

    Returns:
        Dictionary containing evaluation metrics
    """
    results = {
        "total": 0,
        "correct": 0,
        "parse_errors": 0,
        "missing_gt": 0,
        "missing_pred": 0,
        "details": [],
        "distances": [],  # Store distances for valid predictions
        "within_10px": 0,  # Count of clicks within 10px of center
        "within_20px": 0   # Count of clicks within 20px of center
    }
    
    # Find all images that have both ground truth and predictions
    all_images = set(ground_truth.keys()) | set(predictions.keys())
    
    for image_path in all_images:
        gt_data = ground_truth.get(image_path)
        pred_text = predictions.get(image_path)
        
        if not gt_data:
            results["missing_gt"] += 1
            continue
        
        if not pred_text:
            results["missing_pred"] += 1
            continue
        
        results["total"] += 1
        
        # Parse prediction
        coords = parse_pyautogui_click(pred_text)
        
        if coords is None:
            results["parse_errors"] += 1
            results["details"].append({
                "image_path": image_path,
                "id": gt_data["id"],
                "status": "parse_error",
                "prediction": pred_text[:100] if pred_text else None,
                "bbox": gt_data["bbox"],
                "resolution": gt_data["resolution"]
            })
            continue
        
        # Get bbox and resolution
        bbox = gt_data["bbox"]
        resolution = gt_data["resolution"]
        
        if not bbox or not resolution:
            results["parse_errors"] += 1
            results["details"].append({
                "image_path": image_path,
                "id": gt_data["id"],
                "status": "missing_bbox",
                "prediction": pred_text[:100] if pred_text else None,
            })
            continue
        
        # Normalize bbox to target resolution
        norm_bbox = normalize_bbox(
            bbox,
            resolution[0],  # width
            resolution[1],  # height
            target_width,
            target_height
        )
        
        # Check if point is in bbox
        is_correct = point_in_bbox(coords, norm_bbox)
        
        if is_correct:
            results["correct"] += 1
        
        # Calculate distance from predicted click to center of bbox
        bbox_center_coords = bbox_center(norm_bbox)
        distance = euclidean_distance(coords, bbox_center_coords)
        results["distances"].append(distance)
        
        # Count clicks within threshold distances
        if distance <= 10:
            results["within_10px"] += 1
        if distance <= 20:
            results["within_20px"] += 1
        
        results["details"].append({
            "image_path": image_path,
            "id": gt_data["id"],
            "status": "correct" if is_correct else "incorrect",
            "predicted_coords": coords,
            "gt_bbox": bbox,
            "norm_gt_bbox": norm_bbox,
            "bbox_center": bbox_center_coords,
            "distance_to_center": distance,
            "resolution": resolution,
            "prediction": pred_text[:100] if pred_text else None,
        })
    
    # Calculate metrics
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
        results["parse_rate"] = (results["total"] - results["parse_errors"]) / results["total"]
    else:
        results["accuracy"] = 0.0
        results["parse_rate"] = 0.0
    
    # Calculate distance statistics
    if results["distances"]:
        distances = sorted(results["distances"])
        mean = sum(distances) / len(distances)
        n = len(distances)
        if n % 2 == 0:
            median = (distances[n // 2 - 1] + distances[n // 2]) / 2.0
        else:
            median = distances[n // 2]
        
        variance = sum((d - mean) ** 2 for d in distances) / len(distances)
        std = math.sqrt(variance) if len(distances) > 1 else 0.0
        
        # Calculate percentages for threshold metrics
        total_valid = len(distances)
        within_10px_pct = (results["within_10px"] / total_valid) * 100 if total_valid > 0 else 0.0
        within_20px_pct = (results["within_20px"] / total_valid) * 100 if total_valid > 0 else 0.0
        
        results["distance_stats"] = {
            "mean": mean,
            "median": median,
            "min": min(distances),
            "max": max(distances),
            "std": std,
            "within_10px_count": results["within_10px"],
            "within_10px_percentage": within_10px_pct,
            "within_20px_count": results["within_20px"],
            "within_20px_percentage": within_20px_pct
        }
    else:
        results["distance_stats"] = {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "within_10px_count": 0,
            "within_10px_percentage": 0.0,
            "within_20px_count": 0,
            "within_20px_percentage": 0.0
        }
    
    return results


def print_results(results: Dict, verbose: bool = False):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("Grounding Evaluation Results")
    print("="*80)
    print(f"Total samples: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Parse errors: {results['parse_errors']}")
    print(f"Missing ground truth: {results['missing_gt']}")
    print(f"Missing predictions: {results['missing_pred']}")
    print(f"\nAccuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Parse rate: {results['parse_rate']:.4f} ({results['parse_rate']*100:.2f}%)")
    
    # Print distance statistics
    if results.get("distance_stats"):
        stats = results["distance_stats"]
        print(f"\nDistance to Bbox Center (Euclidean):")
        print(f"  Mean: {stats['mean']:.2f} pixels")
        print(f"  Median: {stats['median']:.2f} pixels")
        print(f"  Min: {stats['min']:.2f} pixels")
        print(f"  Max: {stats['max']:.2f} pixels")
        print(f"  Std Dev: {stats['std']:.2f} pixels")
        print(f"\nThreshold Metrics:")
        print(f"  Within 10px: {stats['within_10px_count']} ({stats['within_10px_percentage']:.2f}%)")
        print(f"  Within 20px: {stats['within_20px_count']} ({stats['within_20px_percentage']:.2f}%)")
    
    print("="*80)
    
    if verbose and results["details"]:
        print("\nDetailed Results (first 10):")
        print("-"*80)
        for detail in results["details"][:10]:
            print(f"Image: {detail['image_path']}")
            print(f"  Status: {detail['status']}")
            if detail.get("predicted_coords"):
                print(f"  Predicted: {detail['predicted_coords']}")
            if detail.get("norm_gt_bbox"):
                print(f"  GT Bbox (normalized): {detail['norm_gt_bbox']}")
            if detail.get("bbox_center"):
                print(f"  Bbox Center: {detail['bbox_center']}")
            if detail.get("distance_to_center") is not None:
                print(f"  Distance to Center: {detail['distance_to_center']:.2f} pixels")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate grounding protocol on Wave UI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate predictions from a JSONL file
  python scripts/grounding_eval.py --predictions predictions.jsonl

  # Evaluate with custom paths
  python scripts/grounding_eval.py \\
    --predictions predictions.jsonl \\
    --ground-truth base/prompts/prompts.jsonl

  # Show detailed results
  python scripts/grounding_eval.py --predictions predictions.jsonl --verbose
        """
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="base/prompts/prompts.jsonl",
        help="Path to ground truth prompts.jsonl file"
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1400,
        help="Target normalized width (default: 1400)"
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=800,
        help="Target normalized height (default: 800)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed results for each sample"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results JSON file"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth)
    
    # Check if files exist
    if not predictions_path.exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        return 1
    
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        return 1
    
    # Load data
    print("Loading ground truth...")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(ground_truth)} ground truth records")
    
    print("Loading predictions...")
    predictions = load_predictions(predictions_path)
    print(f"Loaded {len(predictions)} predictions")
    
    # Evaluate
    print("Evaluating...")
    results = evaluate_grounding(
        ground_truth,
        predictions,
        args.target_width,
        args.target_height
    )
    
    # Print results
    print_results(results, args.verbose)
    
    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

