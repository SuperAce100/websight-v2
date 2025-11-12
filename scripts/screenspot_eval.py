import json
import re
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

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
    Parse coordinates from text.

    Args:
        text: Text that may contain coordinates

    Returns:
        Tuple of (x, y) coordinates if found, None otherwise
    """
    # Pattern to match coordinates
    pattern = r'(\d+),\s*(\d+)'
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
        point: (x, y) coordinates
        bbox: [x_min, y_min, x_max, y_max]

    Returns:
        True if point is within bbox, False otherwise
    """
    x_min, y_min, x_max, y_max = bbox
    
    return x_min <= x <= x_max and y_min <= y <= y_max

def main():
    with open("screenspot_pro_predictions_openrouter.jsonl", "r") as f:
        num_correct = 0
        num_total = 0
        for line in f:
            data = json.loads(line)
            output = data["output"]
            x, y = None, None
            if parse_pyautogui_click(output) is None:
                if parse_coordinates(output) is None:
                    print(f"Failed to parse coordinates from {output}")
                    continue
                else:
                    x, y = parse_coordinates(output)
            else:
                x, y = parse_pyautogui_click(output)
            
            bbox = data["bbox"]

            if point_in_bbox(x, y, bbox):
                num_correct += 1
            num_total += 1
        
        print(f"Number of correct predictions: {num_correct}")
        print(f"Number of total predictions: {num_total}")
        print(f"Accuracy: {num_correct / num_total}")

if __name__ == "__main__":
    main()