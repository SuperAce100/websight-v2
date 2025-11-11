#!/usr/bin/env python3
"""
Download and extract the websight-v2 dataset to the cluster storage.

This script downloads the dataset from a specified source and extracts it
to /hai/scratch/websight-v2/data for use in training.

Supported sources:
- Direct URL (HTTP/HTTPS)
- HuggingFace Hub
- Local file copy
"""

import argparse
import os
import sys
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional
import json


def download_progress_hook(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100.0 / total_size, 100.0)
        bar_length = 50
        filled_length = int(bar_length * downloaded // total_size)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        sys.stdout.write(f'\r|{bar}| {percent:.1f}% ({downloaded / 1e9:.2f}GB / {total_size / 1e9:.2f}GB)')
        sys.stdout.flush()
        if downloaded >= total_size:
            print()
    else:
        sys.stdout.write(f'\rDownloaded: {downloaded / 1e6:.1f}MB')
        sys.stdout.flush()


def download_from_url(url: str, output_path: Path) -> Path:
    """
    Download file from URL.
    
    Args:
        url: Download URL
        output_path: Where to save the file
        
    Returns:
        Path to downloaded file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urlretrieve(url, output_path, reporthook=download_progress_hook)
        print(f"✓ Download complete: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def download_from_huggingface(repo_id: str, filename: str, output_path: Path) -> Path:
    """
    Download file from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        filename: File to download from the repo
        output_path: Where to save the file
        
    Returns:
        Path to downloaded file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("✗ huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    
    print(f"Downloading from HuggingFace: {repo_id}/{filename}")
    print(f"Saving to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=output_path.parent,
            local_dir=output_path.parent,
            local_dir_use_symlinks=False
        )
        
        # Move to final location if needed
        if Path(downloaded_path) != output_path:
            shutil.move(downloaded_path, output_path)
        
        print(f"✓ Download complete: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        raise


def extract_archive(archive_path: Path, extract_to: Path) -> Path:
    """
    Extract archive (tar.gz, zip, etc.) to destination.
    
    Args:
        archive_path: Path to archive file
        extract_to: Destination directory
        
    Returns:
        Path to extracted directory
    """
    print(f"\nExtracting: {archive_path}")
    print(f"Destination: {extract_to}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix in ['.gz', '.tgz'] or archive_path.name.endswith('.tar.gz'):
            print("Detected tar.gz archive")
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Show progress
                members = tar.getmembers()
                print(f"Extracting {len(members)} files...")
                for i, member in enumerate(members, 1):
                    tar.extract(member, extract_to)
                    if i % 1000 == 0 or i == len(members):
                        sys.stdout.write(f'\rExtracted: {i}/{len(members)} files')
                        sys.stdout.flush()
                print()
                
        elif archive_path.suffix == '.zip':
            print("Detected zip archive")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                print(f"Extracting {len(members)} files...")
                for i, member in enumerate(members, 1):
                    zip_ref.extract(member, extract_to)
                    if i % 1000 == 0 or i == len(members):
                        sys.stdout.write(f'\rExtracted: {i}/{len(members)} files')
                        sys.stdout.flush()
                print()
                
        elif archive_path.suffix == '.tar':
            print("Detected tar archive")
            with tarfile.open(archive_path, 'r:') as tar:
                members = tar.getmembers()
                print(f"Extracting {len(members)} files...")
                tar.extractall(extract_to, members=members)
                
        else:
            print(f"✗ Unknown archive format: {archive_path.suffix}")
            sys.exit(1)
        
        print(f"✓ Extraction complete")
        return extract_to
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        raise


def verify_dataset(data_dir: Path) -> bool:
    """
    Verify the dataset structure is correct.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        True if valid, False otherwise
    """
    print(f"\nVerifying dataset structure: {data_dir}")
    
    # Check for required files
    prompts_file = data_dir / "prompts.jsonl"
    images_dir = data_dir / "images"
    
    if not prompts_file.exists():
        print(f"✗ Missing required file: prompts.jsonl")
        return False
    
    if not images_dir.exists():
        print(f"✗ Missing required directory: images/")
        return False
    
    if not images_dir.is_dir():
        print(f"✗ images/ is not a directory")
        return False
    
    # Count records in prompts.jsonl
    print("Checking prompts.jsonl...")
    record_count = 0
    try:
        with open(prompts_file, 'r') as f:
            for line in f:
                if line.strip():
                    record_count += 1
                    # Validate first record
                    if record_count == 1:
                        try:
                            data = json.loads(line)
                            if 'original' not in data or 'prompt' not in data:
                                print("✗ Invalid record format in prompts.jsonl")
                                return False
                        except json.JSONDecodeError:
                            print("✗ Invalid JSON in prompts.jsonl")
                            return False
    except Exception as e:
        print(f"✗ Error reading prompts.jsonl: {e}")
        return False
    
    print(f"✓ Found {record_count} records in prompts.jsonl")
    
    # Count images
    print("Checking images directory...")
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    image_count = len(image_files)
    
    if image_count == 0:
        print("✗ No images found in images/")
        return False
    
    print(f"✓ Found {image_count} images")
    
    # Check if counts roughly match (some records might not have images)
    if abs(record_count - image_count) > record_count * 0.1:
        print(f"⚠ Warning: Record count ({record_count}) and image count ({image_count}) differ significantly")
    
    print("\n✓ Dataset verification passed!")
    return True


def copy_local_dataset(source_path: Path, dest_path: Path):
    """
    Copy dataset from local path to destination.
    
    Args:
        source_path: Source directory
        dest_path: Destination directory
    """
    print(f"Copying dataset from: {source_path}")
    print(f"To: {dest_path}")
    
    if not source_path.exists():
        print(f"✗ Source path does not exist: {source_path}")
        sys.exit(1)
    
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy prompts.jsonl
    prompts_src = source_path / "prompts.jsonl"
    if prompts_src.exists():
        print("Copying prompts.jsonl...")
        shutil.copy2(prompts_src, dest_path / "prompts.jsonl")
    
    # Copy images directory
    images_src = source_path / "images"
    images_dest = dest_path / "images"
    if images_src.exists():
        print("Copying images directory...")
        if images_dest.exists():
            shutil.rmtree(images_dest)
        shutil.copytree(images_src, images_dest)
    
    print("✓ Copy complete")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract websight-v2 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from URL
  python scripts/download_dataset.py --url https://example.com/dataset.tar.gz
  
  # Download from HuggingFace
  python scripts/download_dataset.py --hf-repo username/websight-v2 --hf-file dataset.tar.gz
  
  # Copy from local directory
  python scripts/download_dataset.py --local-path /path/to/dataset
  
  # Specify custom destination
  python scripts/download_dataset.py --url https://example.com/dataset.tar.gz --dest /custom/path
        """
    )
    
    parser.add_argument(
        "--url",
        type=str,
        help="Direct download URL"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="HuggingFace repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--hf-file",
        type=str,
        help="File to download from HuggingFace repo"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="Local path to copy dataset from"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/hai/scratch/websight-v2/data",
        help="Destination directory (default: /hai/scratch/websight-v2/data)"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="/tmp/websight-v2-download",
        help="Temporary directory for downloads (default: /tmp/websight-v2-download)"
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep downloaded archive after extraction"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip dataset verification after extraction"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    source_count = sum([
        args.url is not None,
        args.hf_repo is not None,
        args.local_path is not None
    ])
    
    if source_count == 0:
        print("✗ Error: Must specify one source: --url, --hf-repo, or --local-path")
        parser.print_help()
        sys.exit(1)
    
    if source_count > 1:
        print("✗ Error: Can only specify one source at a time")
        sys.exit(1)
    
    if args.hf_repo and not args.hf_file:
        print("✗ Error: --hf-file required when using --hf-repo")
        sys.exit(1)
    
    dest_path = Path(args.dest)
    temp_dir = Path(args.temp_dir)
    
    print("="*80)
    print("WebSight-v2 Dataset Download & Setup")
    print("="*80)
    print(f"Destination: {dest_path}")
    print()
    
    # Check if destination already exists
    if dest_path.exists() and (dest_path / "prompts.jsonl").exists():
        print(f"⚠ Warning: Dataset already exists at {dest_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
        print("Removing existing dataset...")
        shutil.rmtree(dest_path)
    
    try:
        if args.local_path:
            # Copy from local path
            source_path = Path(args.local_path)
            copy_local_dataset(source_path, dest_path)
            
        else:
            # Download from URL or HuggingFace
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            if args.url:
                # Determine filename from URL
                filename = args.url.split('/')[-1]
                if '?' in filename:
                    filename = filename.split('?')[0]
                archive_path = temp_dir / filename
                
                download_from_url(args.url, archive_path)
                
            elif args.hf_repo:
                archive_path = temp_dir / args.hf_file
                download_from_huggingface(args.hf_repo, args.hf_file, archive_path)
            
            # Extract archive
            extract_archive(archive_path, dest_path)
            
            # Clean up archive unless --keep-archive
            if not args.keep_archive:
                print(f"\nCleaning up: {archive_path}")
                archive_path.unlink()
            
            # Clean up temp directory if empty
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        
        # Verify dataset structure
        if not args.skip_verify:
            if not verify_dataset(dest_path):
                print("\n✗ Dataset verification failed!")
                sys.exit(1)
        
        print("\n" + "="*80)
        print("✓ Dataset setup complete!")
        print("="*80)
        print(f"Location: {dest_path}")
        print(f"You can now run: python scripts/transform_for_training.py")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

