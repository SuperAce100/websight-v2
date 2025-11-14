#!/usr/bin/env python3
"""
Utility helpers for preparing the ScreenSpot-Pro benchmark dataset.

This module centralizes the logic for:
  * Downloading the raw dataset snapshot from HuggingFace
  * Copying images into a flat local folder
  * Converting annotations into ShareGPT-style JSONL (messages + bbox metadata)
  * Loading cached records when they already exist on disk

Both the benchmark runner and standalone prep scripts import these helpers to
avoid code duplication (the same logic previously lived inside
`scripts/benchmark_screenspot_openrouter.py`).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "huggingface_hub is required. Install with `pip install huggingface-hub`."
    ) from exc

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in using electronic devices and interacting with "
    "graphic interfaces. You should not call any external tools."
)


@dataclass(frozen=True)
class ScreenSpotPaths:
    """Resolved paths that scripts may reuse."""

    root: Path
    images: Path
    jsonl: Path
    raw: Path


def _resolve_paths(output_dir: os.PathLike[str] | str) -> ScreenSpotPaths:
    root = Path(output_dir).expanduser().resolve()
    images = root / "images"
    jsonl_path = root / "data.jsonl"
    raw_dir = root / "raw"
    return ScreenSpotPaths(root=root, images=images, jsonl=jsonl_path, raw=raw_dir)


def dataset_available(output_dir: os.PathLike[str] | str) -> bool:
    """Return True when `data.jsonl` already exists with at least one record."""
    paths = _resolve_paths(output_dir)
    return paths.jsonl.exists() and paths.jsonl.stat().st_size > 0


def load_cached_records(output_dir: os.PathLike[str] | str) -> List[Dict]:
    """Load cached ScreenSpot-Pro records if they have already been prepared."""
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


def _gather_annotations(annotations_dir: Path) -> List[Dict]:
    annotation_files = sorted(annotations_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation JSON files found under {annotations_dir}"
        )

    entries: List[Dict] = []
    for ann_file in annotation_files:
        with ann_file.open("r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {ann_file}") from exc

        if isinstance(payload, list):
            entries.extend(payload)
        elif isinstance(payload, dict):
            for key in ("annotations", "data", "samples", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    entries.extend(value)
                    break
            else:
                # treat dict as single annotation
                entries.append(payload)
        else:
            raise ValueError(f"Unsupported annotation format in {ann_file}")

    if not entries:
        raise ValueError("Annotation files were empty.")

    return entries


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


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


def prepare_screenspot_pro(
    output_dir: os.PathLike[str] | str = "screenspot_pro",
    refresh: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[Dict], str]:
    """
    Download + transform the ScreenSpot-Pro dataset if needed.

    Args:
        output_dir: Directory where the processed dataset should live.
        refresh: When True, re-download/reprocess even if cached data exist.
        limit: Optional cap on number of processed samples (debugging).

    Returns:
        (records, output_dir_str)
    """

    paths = _resolve_paths(output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.raw.mkdir(parents=True, exist_ok=True)

    if dataset_available(paths.root) and not refresh:
        return load_cached_records(paths.root), str(paths.root)

    print("Downloading ScreenSpot-Pro snapshot from HuggingFace...")
    snapshot_path = Path(
        snapshot_download(
            repo_id="likaixin/ScreenSpot-Pro",
            repo_type="dataset",
            local_dir=paths.raw,
            local_dir_use_symlinks=False,
            allow_patterns=(
                "annotations/*.json",
                "images/**/*.png",
                "images/**/*.jpg",
                "README.md",
                "LICENSE.md",
            ),
        )
    ).expanduser().resolve()

    annotations_dir = snapshot_path / "annotations"
    raw_images_dir = snapshot_path / "images"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations under {annotations_dir}")
    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Missing images under {raw_images_dir}")

    annotations = _gather_annotations(annotations_dir)
    print(f"Loaded {len(annotations)} annotation entries.")

    processed_records: List[Dict] = []
    jsonl_tmp = paths.jsonl.with_suffix(".tmp")

    with jsonl_tmp.open("w", encoding="utf-8") as writer:
        iterator: Iterable[Dict] = annotations
        if limit is not None:
            iterator = annotations[:limit]

        for idx, ann in enumerate(tqdm(iterator, desc="Preparing ScreenSpot-Pro")):
            source_rel = ann.get("img_filename")
            if not source_rel:
                continue

            source_path = raw_images_dir / source_rel
            if not source_path.exists():
                print(f"[WARN] Missing source image: {source_path}")
                continue

            ext = Path(source_rel).suffix or ".png"
            dest_filename = f"{idx:06d}{ext}"
            dest_path = paths.images / dest_filename

            try:
                _copy_image(source_path, dest_path)
            except Exception as exc:
                print(f"[WARN] Failed to copy {source_path}: {exc}")
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
            if isinstance(ann.get("img_size"), (list, tuple)) and len(
                ann["img_size"]
            ) >= 2:
                width, height = ann["img_size"][:2]

            user_text = _format_user_text(instruction, width, height)
            record = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>\n{user_text}"},
                ],
                "image_path": f"images/{dest_filename}",
                "images": [f"images/{dest_filename}"],
                "instruction": instruction,
                "user_text": user_text,
                "sample_id": idx,
            }

            if bbox:
                record["bbox"] = bbox
                record["gt_bbox"] = bbox

            # Preserve remaining metadata except the image filename (already recorded)
            for key, value in ann.items():
                if key in {"img_filename", "bbox"}:
                    continue
                if value is not None:
                    record[key] = value

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_records.append(record)

    jsonl_tmp.replace(paths.jsonl)
    print(f"Wrote {len(processed_records)} samples to {paths.jsonl}")
    return processed_records, str(paths.root)


def ensure_dataset(
    output_dir: os.PathLike[str] | str = "screenspot_pro",
    refresh: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[Dict], str]:
    """
    Convenience wrapper that always returns (records, media_dir).

    This is what downstream scripts should call when they need the dataset.
    """
    start = time.time()
    records, media_dir = prepare_screenspot_pro(
        output_dir=output_dir,
        refresh=refresh,
        limit=limit,
    )
    elapsed = time.time() - start
    print(f"ScreenSpot-Pro ready ({len(records)} samples) in {elapsed:.1f}s")
    return records, media_dir


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ScreenSpotPaths",
    "dataset_available",
    "load_cached_records",
    "prepare_screenspot_pro",
    "ensure_dataset",
]

