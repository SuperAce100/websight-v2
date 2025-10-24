#!/usr/bin/env python3
"""
Download script for AgentNet dataset.

This script downloads the AgentNet dataset files that are expected by scripts/agentnet.py:
- agentnet_ubuntu_5k.jsonl
- agentnet_win_mac_18k.jsonl

The script will create a data directory and download the files there.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tqdm import tqdm

console = Console()

# Dataset URLs - these would need to be updated with actual URLs
# For now, using placeholder URLs that would need to be replaced with real ones
DATASET_URLS = {
    "agentnet_ubuntu_5k.jsonl": "https://huggingface.co/datasets/agentnet/agentnet-ubuntu-5k/resolve/main/agentnet_ubuntu_5k.jsonl",
    "agentnet_win_mac_18k.jsonl": "https://huggingface.co/datasets/agentnet/agentnet-win-mac-18k/resolve/main/agentnet_win_mac_18k.jsonl",
}

# Alternative URLs if the above don't work (these are examples)
FALLBACK_URLS = {
    "agentnet_ubuntu_5k.jsonl": "https://example.com/agentnet_ubuntu_5k.jsonl",
    "agentnet_win_mac_18k.jsonl": "https://example.com/agentnet_win_mac_18k.jsonl",
}


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to the specified path with progress bar.
    
    Args:
        url: URL to download from
        filepath: Local path to save the file
        chunk_size: Size of chunks to read at a time
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=filepath.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error downloading {url}: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Unexpected error downloading {url}: {e}[/red]")
        return False


def verify_jsonl_file(filepath: Path) -> bool:
    """
    Verify that the downloaded file is a valid JSONL file.
    
    Args:
        filepath: Path to the file to verify
        
    Returns:
        True if file is valid JSONL, False otherwise
    """
    try:
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            # Check first few lines to ensure they're valid JSON
            for i, line in enumerate(f):
                if i >= 5:  # Only check first 5 lines
                    break
                line = line.strip()
                if line:
                    json.loads(line)
        
        console.print(f"[green]✓ {filepath.name} is valid JSONL[/green]")
        return True
        
    except json.JSONDecodeError as e:
        console.print(f"[red]✗ {filepath.name} is not valid JSONL: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Error verifying {filepath.name}: {e}[/red]")
        return False


def download_dataset(
    data_dir: Path,
    use_fallback: bool = False,
    verify_files: bool = True
) -> bool:
    """
    Download the AgentNet dataset files.
    
    Args:
        data_dir: Directory to save the dataset files
        use_fallback: Whether to use fallback URLs if primary URLs fail
        verify_files: Whether to verify downloaded files are valid JSONL
        
    Returns:
        True if all downloads successful, False otherwise
    """
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    urls_to_use = FALLBACK_URLS if use_fallback else DATASET_URLS
    
    console.print(f"[bold]Downloading AgentNet dataset to {data_dir}[/bold]")
    console.print(f"Using {'fallback' if use_fallback else 'primary'} URLs")
    
    success_count = 0
    total_files = len(urls_to_use)
    
    for filename, url in urls_to_use.items():
        filepath = data_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            console.print(f"[yellow]Skipping {filename} (already exists)[/yellow]")
            if verify_files and verify_jsonl_file(filepath):
                success_count += 1
            continue
        
        console.print(f"[blue]Downloading {filename}...[/blue]")
        
        if download_file(url, filepath):
            if verify_files:
                if verify_jsonl_file(filepath):
                    success_count += 1
                else:
                    # Remove invalid file
                    filepath.unlink()
            else:
                success_count += 1
        else:
            console.print(f"[red]Failed to download {filename}[/red]")
    
    if success_count == total_files:
        console.print(f"[green]✓ Successfully downloaded all {total_files} files[/green]")
        return True
    else:
        console.print(f"[red]✗ Only {success_count}/{total_files} files downloaded successfully[/red]")
        return False


def main():
    """Main function to handle command line arguments and download dataset."""
    parser = argparse.ArgumentParser(
        description="Download AgentNet dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_agentnet.py
  python scripts/download_agentnet.py --data-dir ./data
  python scripts/download_agentnet.py --use-fallback
  python scripts/download_agentnet.py --no-verify
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save dataset files (default: ./data)"
    )
    
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback URLs if primary URLs fail"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of downloaded files"
    )
    
    parser.add_argument(
        "--list-urls",
        action="store_true",
        help="List the URLs that would be used for downloading"
    )
    
    args = parser.parse_args()
    
    if args.list_urls:
        urls_to_use = FALLBACK_URLS if args.use_fallback else DATASET_URLS
        console.print("[bold]Dataset URLs:[/bold]")
        for filename, url in urls_to_use.items():
            console.print(f"  {filename}: {url}")
        return
    
    # Check if data directory is writable
    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        test_file = args.data_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        console.print(f"[red]Error: Cannot write to {args.data_dir}: {e}[/red]")
        sys.exit(1)
    
    success = download_dataset(
        data_dir=args.data_dir,
        use_fallback=args.use_fallback,
        verify_files=not args.no_verify
    )
    
    if success:
        console.print(f"\n[bold green]Dataset ready![/bold green]")
        console.print(f"Files saved to: {args.data_dir.absolute()}")
        console.print("\nYou can now run the analysis script:")
        console.print(f"python scripts/agentnet.py")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]Download failed![/bold red]")
        console.print("Try using --use-fallback flag or check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()