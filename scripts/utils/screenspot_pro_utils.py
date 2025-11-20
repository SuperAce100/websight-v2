#!/usr/bin/env python3
"""
Utilities for syncing the ScreenSpot-Pro dataset layout used by the benchmark scripts.

The processed format expected by `run_screenspot_benchmark.py` is:
    screenspot_pro/
      ├── data.jsonl
      └── images/
            000000.png
            000001.png
            ...

This module can:
  * Verify that the processed dataset already exists (load_cached_records)
  * Rebuild the processed dataset from a raw snapshot that still has
    per-application subfolders (rebuild_from_raw)
  * Provide a single `ensure_dataset` entry point used by CLI + Slurm flows
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in using electronic devices and interacting with "
    "graphic interfaces. You should not call any external tools."
)


@dataclass(frozen=True)
class ScreenSpotPaths:
    root: Path
    images: Path
    jsonl: Path
    raw: Path


def _resolve_paths(output_dir: os.PathLike[str] | str) -> ScreenSpotPaths:
    root = Path(output_dir).expanduser().resolve()
    return ScreenSpotPaths(
        root=root,
        images=root / "images",
        jsonl=root / "data.jsonl",
        raw=root / "raw",
    )


def dataset_available(output_dir: os.PathLike[str] | str) -> bool:
    paths = _resolve_paths(output_dir)
    return paths.jsonl.exists() and paths.jsonl.stat().st_size > 0


def load_cached_records(output_dir: os.PathLike[str] | str) -> List[Dict]:
    paths = _resolve_paths(output_dir)
    if not paths.jsonl.exists():
        raise FileNotFoundError(f"Dataset not prepared yet: {paths.jsonl}")

    records: List[Dict] = []
    with paths.jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_user_text(
    instruction: str, width: Optional[int], height: Optional[int]
) -> str:
    size_line = ""
    if width and height:
        size_line = f"Image size: {width}x{height} (width x height).\n"

    return (
        f"Query: {instruction}\n"
        f"{size_line}"
        "Return coordinates in ORIGINAL pixels of the image shown.\n"
        "Output only the coordinate of one point in your response as pyautogui commands.\n"
        "Format: pyautogui.click(x, y)\n"
    )


def _gather_annotations(annotations_dir: Path) -> List[Dict]:
    files = sorted(annotations_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation JSON files under {annotations_dir}")

    entries: List[Dict] = []
    for ann_file in files:
        with ann_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if isinstance(payload, list):
            entries.extend(payload)
        elif isinstance(payload, dict):
            for key in ("annotations", "data", "samples", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    entries.extend(value)
                    break
            else:
                entries.append(payload)
        else:
            raise ValueError(f"Unsupported annotation format in {ann_file}")

    if not entries:
        raise ValueError("Annotation files were empty.")

    return entries


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def rebuild_from_raw(raw_dir: Path, output_dir: Path) -> List[Dict]:
    annotations_dir = raw_dir / "annotations"
    raw_images_dir = raw_dir / "images"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")
    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {raw_images_dir}")

    annotations = _gather_annotations(annotations_dir)

    # Reset processed layout
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    tmp_jsonl = (output_dir / "data.jsonl").with_suffix(".tmp")
    records: List[Dict] = []

    with tmp_jsonl.open("w", encoding="utf-8") as writer:
        for idx, ann in enumerate(tqdm(annotations, desc="Flattening ScreenSpot-Pro")):
            img_filename = ann.get("img_filename")
            if not img_filename:
                continue

            source_path = raw_images_dir / img_filename
            if not source_path.exists():
                continue

            ext = Path(img_filename).suffix or ".png"
            dest_name = f"{idx:06d}{ext}"
            dest_path = images_dir / dest_name

            try:
                _copy_image(source_path, dest_path)
            except Exception:
                continue

            instruction = (
                ann.get("instruction")
                or ann.get("text")
                or ann.get("prompt")
                or "click on the target element"
            )

            bbox = ann.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                bbox = [int(coord) for coord in bbox[:4]]
            else:
                bbox = None

            width = None
            height = None
            if isinstance(ann.get("img_size"), (list, tuple)) and len(ann["img_size"]) >= 2:
                width, height = ann["img_size"][:2]

            user_text = _format_user_text(instruction, width, height)
            record = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>\n{user_text}"},
                ],
                "image_path": f"images/{dest_name}",
                "images": [f"images/{dest_name}"],
                "instruction": instruction,
                "user_text": user_text,
                "sample_id": idx,
            }

            if bbox:
                record["bbox"] = bbox
                record["gt_bbox"] = bbox

            for key, value in ann.items():
                if key in {"img_filename", "bbox"}:
                    continue
                if value is not None:
                    record[key] = value

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    tmp_jsonl.replace(output_dir / "data.jsonl")
    return records


def ensure_dataset(
    output_dir: os.PathLike[str] | str = "screenspot_pro",
    limit: Optional[int] = None,
    raw_source_dir: Optional[os.PathLike[str] | str] = None,
    rebuild_if_missing: bool = True,
    force_rebuild: bool = False,
) -> Tuple[List[Dict], str]:
    paths = _resolve_paths(output_dir)

    needs_rebuild = force_rebuild or (not dataset_available(paths.root) and rebuild_if_missing)
    if needs_rebuild:
        raw_dir = (
            Path(raw_source_dir).expanduser().resolve()
            if raw_source_dir is not None
            else paths.raw
        )
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Cannot rebuild ScreenSpot-Pro dataset without raw snapshot. "
                f"Expected {raw_dir} to exist."
            )
        records = rebuild_from_raw(raw_dir, paths.root)
    else:
        if not dataset_available(paths.root):
            raise FileNotFoundError(
                f"ScreenSpot-Pro dataset not found at {paths.root}. Expected "
                f"{paths.jsonl} to exist with matching images/ directory."
            )
        records = load_cached_records(paths.root)

    if limit is not None:
        records = records[:limit]

    return records, str(paths.root)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ScreenSpotPaths",
    "dataset_available",
    "load_cached_records",
    "rebuild_from_raw",
    "ensure_dataset",
]
