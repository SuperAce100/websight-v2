#!/usr/bin/env python3
"""
CLI wrapper around `screenspot_pro_utils` to materialize the ScreenSpot-Pro dataset.

Usage:
    python scripts/prepare_screenspot_pro.py \
        --output-dir screenspot_pro \
        [--refresh] \
        [--limit 100]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from screenspot_pro_utils import ensure_dataset, dataset_available, load_cached_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and transform the ScreenSpot-Pro dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="screenspot_pro",
        help="Destination folder for processed dataset.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-downloading/re-processing even if cache exists.",
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

    if dataset_available(args.output_dir) and not args.refresh:
        print(
            f"Dataset already prepared at {args.output_dir}. "
            "Use --refresh to rebuild."
        )
    records, media_dir = ensure_dataset(
        output_dir=args.output_dir,
        refresh=args.refresh,
        limit=args.limit,
    )

    print(
        f"\n✓ ScreenSpot-Pro dataset ready with {len(records)} samples "
        f"in {media_dir}"
    )

    if args.summary:
        print_summary(media_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

