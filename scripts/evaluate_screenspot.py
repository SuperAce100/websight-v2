#!/usr/bin/env python3
"""
Evaluate ScreenSpot-Pro predictions against ground truth.

This script computes accuracy and other metrics for model predictions on the
ScreenSpot-Pro benchmark by comparing predicted click coordinates against
ground truth bounding boxes.

Usage:
    python scripts/evaluate_screenspot.py \\
        --predictions runs/screenspot_pro/predictions.jsonl \\
        --ground-truth screenspot_pro/data.jsonl
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def parse_coordinates(text: str) -> Optional[Tuple[int, int]]:
    """
    Parse coordinates from text (fallback parser).
    
    Args:
        text: Text that may contain coordinates
    
    Returns:
        Tuple of (x, y) coordinates if found, None otherwise
    """
    # Pattern to match coordinates like "x, y" or "(x, y)"
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


def distance_to_bbox(x: float, y: float, bbox: List[float]) -> float:
    """
    Calculate minimum distance from a point to a bounding box.
    
    Args:
        x: X coordinate
        y: Y coordinate
        bbox: [x_min, y_min, x_max, y_max]
    
    Returns:
        Minimum distance in pixels (0 if point is inside bbox)
    """
    x_min, y_min, x_max, y_max = bbox
    
    # If point is inside bbox, distance is 0
    if point_in_bbox(x, y, bbox):
        return 0.0
    
    # Calculate distance to nearest edge
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    
    return math.sqrt(dx * dx + dy * dy)


def load_predictions(predictions_file: str) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def load_ground_truth(ground_truth_file: str) -> Dict[int, Dict]:
    """Load ground truth from JSONL file, indexed by sample_id."""
    ground_truth = {}
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                sample_id = record.get("sample_id")
                if sample_id is not None:
                    ground_truth[sample_id] = record
    return ground_truth


def evaluate_predictions(
    predictions: List[Dict],
    ground_truth: Dict[int, Dict],
    verbose: bool = False
) -> Dict:
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions: List of prediction records
        ground_truth: Dict mapping sample_id to ground truth records
        verbose: Print detailed results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    results = {
        "total": 0,
        "parsed": 0,
        "unparsed": 0,
        "correct": 0,
        "incorrect": 0,
        "missing_gt": 0,
        "distances": [],
        "by_application": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_platform": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_ui_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_group": defaultdict(lambda: {"total": 0, "correct": 0}),
    }
    
    for pred in predictions:
        results["total"] += 1
        sample_id = pred.get("sample_id")
        output = pred.get("output", "")
        
        # Parse coordinates from output
        coords = parse_pyautogui_click(output)
        if coords is None:
            coords = parse_coordinates(output)
        
        if coords is None:
            results["unparsed"] += 1
            if verbose:
                print(f"Sample {sample_id}: Failed to parse coordinates from: {output}")
            continue
        
        results["parsed"] += 1
        x, y = coords
        
        # Get ground truth bbox
        # First try from prediction record itself
        bbox = pred.get("bbox") or pred.get("gt_bbox")
        
        # If not in prediction, look up in ground truth
        if bbox is None and sample_id is not None:
            gt_record = ground_truth.get(sample_id)
            if gt_record:
                bbox = gt_record.get("bbox") or gt_record.get("gt_bbox")
        
        if bbox is None:
            results["missing_gt"] += 1
            if verbose:
                print(f"Sample {sample_id}: No ground truth bbox found")
            continue
        
        # Check if prediction is correct
        is_correct = point_in_bbox(x, y, bbox)
        distance = distance_to_bbox(x, y, bbox)
        
        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1
        
        results["distances"].append(distance)
        
        # Track by metadata categories
        for key, category_key in [
            ("application", "by_application"),
            ("platform", "by_platform"),
            ("ui_type", "by_ui_type"),
            ("group", "by_group")
        ]:
            value = pred.get(key)
            if value:
                results[category_key][value]["total"] += 1
                if is_correct:
                    results[category_key][value]["correct"] += 1
        
        if verbose and not is_correct:
            print(f"Sample {sample_id}: Incorrect - predicted ({x}, {y}), bbox {bbox}, distance {distance:.1f}px")
    
    return results


def print_results(results: Dict, output_json: Optional[str] = None):
    """Print evaluation results in a formatted table."""
    print()
    print("=" * 80)
    print("ScreenSpot-Pro Evaluation Results")
    print("=" * 80)
    print()
    
    # Overall metrics
    print("Overall Metrics:")
    print("-" * 80)
    total = results["total"]
    parsed = results["parsed"]
    correct = results["correct"]
    
    print(f"  Total samples:        {total}")
    print(f"  Parsed successfully:  {parsed} ({parsed/total*100:.1f}%)")
    print(f"  Failed to parse:      {results['unparsed']} ({results['unparsed']/total*100:.1f}%)")
    print(f"  Missing ground truth: {results['missing_gt']}")
    print()
    
    evaluated = correct + results["incorrect"]
    if evaluated > 0:
        accuracy = correct / evaluated * 100
        print(f"  ✓ Correct:            {correct} / {evaluated} ({accuracy:.2f}%)")
        print(f"  ✗ Incorrect:          {results['incorrect']} / {evaluated} ({100-accuracy:.2f}%)")
    else:
        print("  No samples evaluated (all failed to parse or missing ground truth)")
    
    # Distance statistics
    if results["distances"]:
        distances = results["distances"]
        avg_distance = sum(distances) / len(distances)
        median_distance = sorted(distances)[len(distances) // 2]
        print()
        print(f"  Average distance:     {avg_distance:.1f} pixels")
        print(f"  Median distance:      {median_distance:.1f} pixels")
    
    print()
    
    # Breakdown by categories
    for category_name, category_key in [
        ("Application", "by_application"),
        ("Platform", "by_platform"),
        ("UI Type", "by_ui_type"),
        ("Group", "by_group")
    ]:
        category_data = results[category_key]
        if category_data:
            print(f"Breakdown by {category_name}:")
            print("-" * 80)
            
            # Sort by total count descending
            sorted_items = sorted(
                category_data.items(),
                key=lambda x: x[1]["total"],
                reverse=True
            )
            
            for name, stats in sorted_items:
                total = stats["total"]
                correct = stats["correct"]
                accuracy = correct / total * 100 if total > 0 else 0
                print(f"  {name:30s} {correct:4d} / {total:4d} ({accuracy:5.1f}%)")
            
            print()
    
    print("=" * 80)
    
    # Save JSON results if requested
    if output_json:
        # Convert defaultdicts to regular dicts for JSON serialization
        json_results = {
            "total": results["total"],
            "parsed": results["parsed"],
            "unparsed": results["unparsed"],
            "correct": results["correct"],
            "incorrect": results["incorrect"],
            "missing_gt": results["missing_gt"],
            "accuracy": correct / evaluated * 100 if evaluated > 0 else 0,
            "avg_distance": sum(results["distances"]) / len(results["distances"]) if results["distances"] else 0,
            "median_distance": sorted(results["distances"])[len(results["distances"]) // 2] if results["distances"] else 0,
            "by_application": dict(results["by_application"]),
            "by_platform": dict(results["by_platform"]),
            "by_ui_type": dict(results["by_ui_type"]),
            "by_group": dict(results["by_group"]),
        }
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_json}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ScreenSpot-Pro predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_screenspot.py \\
    --predictions runs/screenspot_pro/predictions.jsonl \\
    --ground-truth screenspot_pro/data.jsonl
  
  # With verbose output and JSON export
  python scripts/evaluate_screenspot.py \\
    --predictions runs/screenspot_pro/predictions.jsonl \\
    --ground-truth screenspot_pro/data.jsonl \\
    --verbose \\
    --output-json results.json
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
        required=True,
        help="Path to ground truth JSONL file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each sample"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.predictions).exists():
        print(f"✗ Error: Predictions file not found: {args.predictions}")
        return 1
    
    if not Path(args.ground_truth).exists():
        print(f"✗ Error: Ground truth file not found: {args.ground_truth}")
        return 1
    
    print("Loading predictions and ground truth...")
    
    try:
        predictions = load_predictions(args.predictions)
        print(f"  Loaded {len(predictions)} predictions")
    except Exception as e:
        print(f"✗ Error loading predictions: {e}")
        return 1
    
    try:
        ground_truth = load_ground_truth(args.ground_truth)
        print(f"  Loaded {len(ground_truth)} ground truth records")
    except Exception as e:
        print(f"✗ Error loading ground truth: {e}")
        return 1
    
    if not predictions:
        print("✗ Error: No predictions found")
        return 1
    
    # Evaluate
    print("Evaluating predictions...")
    results = evaluate_predictions(predictions, ground_truth, args.verbose)
    
    # Print results
    print_results(results, args.output_json)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

