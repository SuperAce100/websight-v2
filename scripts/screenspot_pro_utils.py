#!/usr/bin/env python3
"""
Utilities for working with a locally prepared ScreenSpot-Pro dataset.

The workflow now assumes images + `data.jsonl` already exist on disk (e.g.,
copied to /hai/scratch/... on the cluster). These helpers simply verify that
structure and load cached records so inference scripts stay consistent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def ensure_dataset(
    output_dir: Path | str = "screenspot_pro",
    limit: Optional[int] = None,
) -> Tuple[List[Dict], str]:
    """
    Load cached ScreenSpot-Pro records from an existing directory.

    Args:
        output_dir: Directory containing `data.jsonl` and `images/`.
        limit: Optional cap on number of records returned.
    """
    paths = _resolve_paths(output_dir)

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
    "ensure_dataset",
]

