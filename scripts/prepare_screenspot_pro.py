#!/usr/bin/env python3
"""
Prepare ScreenSpot-Pro dataset for benchmarking.

Downloads the ScreenSpot-Pro dataset from HuggingFace and transforms it to a format
suitable for model inference and evaluation. Can be run standalone or as part of a
larger workflow.

Usage:
    python scripts/prepare_screenspot_pro.py --output-dir screenspot_pro
    python scripts/prepare_screenspot_pro.py --skip-download  # Load existing dataset
"""

import argparse
import sys
from pathlib import Path

from screenspot_pro_utils import download_screenspot_pro, load_screenspot_pro


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ScreenSpot-Pro dataset for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and prepare dataset
  python scripts/prepare_screenspot_pro.py --output-dir screenspot_pro
  
  # Load existing dataset (skip download)
  python scripts/prepare_screenspot_pro.py --skip-download --output-dir screenspot_pro
  
  # Download with custom retry settings
  python scripts/prepare_screenspot_pro.py --max-retries 10 --retry-delay 120
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="screenspot_pro",
        help="Output directory for processed dataset (data.jsonl) (default: screenspot_pro)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory to save images (default: output-dir/images). Use this to store images separately, e.g., /hai/scratch/asanshay/screenspot-pro/images"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if dataset already exists"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts for rate limiting (default: 5)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=60,
        help="Initial delay in seconds between retries, uses exponential backoff (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Check if dataset already exists
    data_jsonl = Path(args.output_dir) / "data.jsonl"
    
    if args.skip_download and data_jsonl.exists():
        print("=" * 80)
        print("Loading existing ScreenSpot-Pro dataset")
        print("=" * 80)
        
        # Check if images directory exists
        if args.images_dir:
            images_dir = Path(args.images_dir)
        else:
            images_dir = Path(args.output_dir) / "images"
        
        if not images_dir.exists():
            print(f"\n⚠️  Warning: data.jsonl exists but images directory not found at {images_dir}")
            print("  Re-downloading dataset to ensure completeness...")
        else:
            try:
                records, media_dir = load_screenspot_pro(args.output_dir, args.images_dir)
                
                # Count images
                image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
                image_count = len(image_files)
                
                print(f"\n✓ Successfully loaded {len(records)} records from {data_jsonl}")
                print(f"  Media directory: {media_dir}")
                print(f"  Images found: {image_count}")
                
                # Verify image count
                if image_count < len(records) - 10:
                    print(f"\n⚠️  Warning: Image count ({image_count}) is significantly less than record count ({len(records)})")
                    print("  Some images may be missing. Consider re-downloading without --skip-download")
                
                return 0
            except Exception as e:
                print(f"\n✗ Error loading dataset: {e}")
                return 1
    
    # Download and prepare dataset
    print("=" * 80)
    print("Downloading and preparing ScreenSpot-Pro dataset")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Max retries: {args.max_retries}")
    print(f"Retry delay: {args.retry_delay}s (exponential backoff)")
    print()
    
    try:
        records, media_dir = download_screenspot_pro(
            output_dir=args.output_dir,
            images_dir=args.images_dir,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        # Verify images were created
        if args.images_dir:
            images_dir = Path(args.images_dir)
        else:
            images_dir = Path(args.output_dir) / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            image_count = len(image_files)
        else:
            image_count = 0
        
        print()
        print("=" * 80)
        print("✓ Dataset preparation complete!")
        print("=" * 80)
        print(f"  Records: {len(records)}")
        print(f"  Images: {image_count}")
        print(f"  Data file: {data_jsonl}")
        print(f"  Media directory: {media_dir}")
        
        # Warn if image count doesn't match
        if image_count < len(records) - 10:
            print()
            print(f"⚠️  Warning: Image count ({image_count}) is less than record count ({len(records)})")
            print("  Some images may have failed to download or copy.")
        
        print()
        print("Next steps:")
        print("  1. Run benchmark:")
        print(f"     python scripts/run_screenspot_benchmark.py \\")
        print(f"       --model-name-or-path Asanshay/websight-v2-grounded \\")
        print(f"       --data-dir {args.output_dir}")
        print("  2. Or submit Slurm job:")
        print("     sbatch slurm/screenspot_benchmark.slurm")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Error preparing dataset: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

