#!/usr/bin/env python3
"""
Validate or summarize an existing ScreenSpot-Pro dataset directory.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))

from screenspot_pro_utils import dataset_available, ensure_dataset, load_cached_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify ScreenSpot-Pro dataset presence and print a summary."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="screenspot_pro",
        help="Destination folder for processed dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples (useful for debugging).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a short summary after preparation completes.",
    )
    parser.add_argument(
        "--sync-from-raw",
        type=str,
        default=None,
        help="Path to raw ScreenSpot-Pro snapshot (annotations/ + images/) to rebuild from.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding processed data even if it already exists.",
    )
    return parser.parse_args()


def print_summary(output_dir: str) -> None:
    jsonl_path = Path(output_dir) / "data.jsonl"
    if not jsonl_path.exists():
        print(f"No summary available: {jsonl_path} missing.")
        return

    records = load_cached_records(output_dir)
    total = len(records)
    bbox = sum(1 for rec in records if rec.get("gt_bbox"))
    apps = {}
    for rec in records:
        app = rec.get("application", "unknown")
        apps[app] = apps.get(app, 0) + 1

    print("\n=== ScreenSpot-Pro Summary ===")
    print(f"Records: {total}")
    print(f"Records with bbox: {bbox}")
    print(f"Unique applications: {len(apps)}")
    top_apps = sorted(apps.items(), key=lambda pair: pair[1], reverse=True)[:5]
    for name, count in top_apps:
        print(f"  - {name}: {count}")


def main() -> int:
    args = parse_args()

    records, media_dir = ensure_dataset(
        output_dir=args.output_dir,
        limit=args.limit,
        raw_source_dir=args.sync_from_raw,
        rebuild_if_missing=True,
        force_rebuild=args.force_rebuild,
    )

    print(
        f"\n✓ ScreenSpot-Pro dataset verified with {len(records)} samples "
        f"in {media_dir}"
    )

    if args.summary:
        print_summary(media_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

