#!/usr/bin/env python3
"""
Evaluate ScreenSpot-Pro predictions by checking whether clicks fall inside GT boxes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_pyautogui_click(text: str) -> Optional[Tuple[int, int]]:
    pattern = r"pyautogui\.click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def parse_fallback_coordinates(text: str) -> Optional[Tuple[int, int]]:
    pattern = r"(\d+)\s*,\s*(\d+)"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def build_bbox_index(dataset_records: Iterable[Dict]) -> Dict[int, List[float]]:
    index: Dict[int, List[float]] = {}
    for rec in dataset_records:
        bbox = rec.get("gt_bbox") or rec.get("bbox")
        sample_id = rec.get("sample_id")
        if bbox and sample_id is not None:
            index[int(sample_id)] = bbox
    return index


def extract_coordinates(output_text: str) -> Optional[Tuple[int, int]]:
    parsed = parse_pyautogui_click(output_text)
    if parsed:
        return parsed
    return parse_fallback_coordinates(output_text)


def evaluate(
    predictions_path: Path,
    dataset_path: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, float]:
    predictions = load_jsonl(predictions_path)
    bbox_index: Dict[int, List[float]] = {}

    if dataset_path and dataset_path.exists():
        bbox_index = build_bbox_index(load_jsonl(dataset_path))

    total = 0
    parsed = 0
    hits = 0
    distances: List[float] = []

    for pred in predictions:
        total += 1
        coords = extract_coordinates(pred.get("output", ""))
        if not coords:
            if not quiet:
                print(f"[WARN] Could not parse coordinates: {pred.get('output')}")
            continue

        parsed += 1
        bbox = (
            pred.get("gt_bbox")
            or pred.get("bbox")
            or bbox_index.get(int(pred.get("sample_id", -1)))
        )

        if not bbox:
            if not quiet:
                print(
                    f"[WARN] Missing bbox for sample {pred.get('sample_id')} "
                    f"(prediction skipped)."
                )
            continue

        if point_in_bbox(coords, bbox):
            hits += 1

        center = bbox_center(bbox)
        distances.append(euclidean_distance(coords, center))

    accuracy = hits / parsed if parsed else 0.0
    mean_distance = sum(distances) / len(distances) if distances else float("nan")

    summary = {
        "total_predictions": total,
        "parsed_predictions": parsed,
        "hits": hits,
        "accuracy": accuracy,
        "mean_distance_to_center": mean_distance,
    }

    if not quiet:
        print("\n=== ScreenSpot-Pro Evaluation ===")
        print(f"Predictions file: {predictions_path}")
        if dataset_path:
            print(f"Dataset file: {dataset_path}")
        print(f"Total predictions: {total}")
        print(f"Parsed predictions: {parsed}")
        print(f"Hits: {hits}")
        print(f"Accuracy: {accuracy:.2%}")
        if distances:
            print(f"Mean distance to bbox center: {mean_distance:.2f} px")
        else:
            print("Mean distance to bbox center: N/A")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ScreenSpot-Pro predictions (pyautogui.click outputs)."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="screenspot_pro/data.jsonl",
        help="Ground truth dataset JSONL (default: screenspot_pro/data.jsonl).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump metrics as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample warnings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions).expanduser()
    dataset_path = Path(args.dataset).expanduser()
    summary = evaluate(
        predictions_path=predictions_path,
        dataset_path=dataset_path if dataset_path.exists() else None,
        quiet=args.quiet,
    )

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nMetrics saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())