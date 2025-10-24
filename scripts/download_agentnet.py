#!/usr/bin/env python3
"""
Download script for AgentNet dataset using Hugging Face datasets library.

This script downloads the AgentNet dataset files using the Hugging Face datasets library
and saves them in the format expected by scripts/agentnet.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Dataset configurations - using available datasets
DATASET_CONFIGS = {
    "agentnet_clicks": {
        "dataset_name": "mlfoundations-cua-dev/agentnet-clicks",
        "output_filename": "agentnet_ubuntu_5k.jsonl",
        "description": "AgentNet Clicks dataset (Ubuntu subset)"
    },
    "agentnet_gimp": {
        "dataset_name": "mlfoundations-cua-dev/agentnet-gimp-trajectories", 
        "output_filename": "agentnet_win_mac_18k.jsonl",
        "description": "AgentNet GIMP trajectories dataset (Windows/Mac subset)"
    }
}

# Sample data generator for testing
def generate_sample_data(output_path: Path, num_samples: int = 100) -> bool:
    """
    Generate sample AgentNet data for testing purposes.
    
    Args:
        output_path: Path to save the sample data
        num_samples: Number of sample records to generate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        console.print(f"[blue]Generating {num_samples} sample records for {output_path.name}...[/blue]")
        
        sample_data = []
        for i in range(num_samples):
            # Generate sample trajectory data that matches the expected format
            trajectory = []
            for step in range(3, 8):  # Random trajectory length between 3-7 steps
                trajectory.append({
                    "index": step,
                    "image": f"step_{step}.png",
                    "value": {
                        "observation": f"Step {step} observation: User is on a webpage",
                        "thought": f"I need to {['click', 'type', 'scroll'][step % 3]} something",
                        "action": f"action_{step}",
                        "code": f"// Step {step} code",
                        "last_step_correct": step % 3 == 0,
                        "last_step_redundant": step % 5 == 0,
                        "reflection": f"Step {step} reflection"
                    }
                })
            
            record = {
                "task_id": f"task_{i:04d}",
                "instruction": f"Sample task {i}: Navigate to a specific webpage and perform actions",
                "natural_language_task": f"Complete task {i} by following the instructions",
                "actual_task": f"Task {i} implementation",
                "task_completed": i % 4 != 0,  # 75% completion rate
                "alignment_score": 0.7 + (i % 30) / 100.0,  # 0.7-0.99
                "efficiency_score": 0.6 + (i % 40) / 100.0,  # 0.6-0.99
                "task_difficulty": (i % 10) + 1,  # 1-10
                "traj": trajectory
            }
            sample_data.append(record)
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in sample_data:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        console.print(f"[green]✓ Successfully generated {len(sample_data)} sample records[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Error generating sample data: {e}[/red]")
        return False


def download_and_save_dataset(
    dataset_name: str,
    output_path: Path,
    description: str
) -> bool:
    """
    Download a dataset from Hugging Face and save as JSONL.
    
    Args:
        dataset_name: Hugging Face dataset name
        output_path: Path to save the JSONL file
        description: Description of the dataset for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        console.print(f"[blue]Downloading {description}...[/blue]")
        
        # Import datasets here to avoid issues if not available
        from datasets import load_dataset
        
        # Load dataset from Hugging Face
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Loading {dataset_name}", total=None)
            
            dataset = load_dataset(dataset_name, split="train")
            progress.update(task, description=f"Loaded {len(dataset)} records")
        
        # Save as JSONL
        console.print(f"[blue]Saving to {output_path}...[/blue]")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in dataset:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        console.print(f"[green]✓ Successfully saved {len(dataset)} records to {output_path.name}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Error downloading {dataset_name}: {e}[/red]")
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


def download_datasets(
    data_dir: Path,
    datasets_to_download: Optional[List[str]] = None,
    verify_files: bool = True,
    use_sample_data: bool = False
) -> bool:
    """
    Download the AgentNet datasets.
    
    Args:
        data_dir: Directory to save the dataset files
        datasets_to_download: List of dataset keys to download (None for all)
        verify_files: Whether to verify downloaded files are valid JSONL
        use_sample_data: Whether to generate sample data instead of downloading
        
    Returns:
        True if all downloads successful, False otherwise
    """
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if datasets_to_download is None:
        datasets_to_download = list(DATASET_CONFIGS.keys())
    
    if use_sample_data:
        console.print(f"[bold]Generating sample AgentNet datasets in {data_dir}[/bold]")
    else:
        console.print(f"[bold]Downloading AgentNet datasets to {data_dir}[/bold]")
    
    success_count = 0
    total_datasets = len(datasets_to_download)
    
    for dataset_key in datasets_to_download:
        if dataset_key not in DATASET_CONFIGS:
            console.print(f"[red]Unknown dataset: {dataset_key}[/red]")
            continue
            
        config = DATASET_CONFIGS[dataset_key]
        output_path = data_dir / config["output_filename"]
        
        # Skip if file already exists
        if output_path.exists():
            console.print(f"[yellow]Skipping {config['output_filename']} (already exists)[/yellow]")
            if verify_files and verify_jsonl_file(output_path):
                success_count += 1
            continue
        
        if use_sample_data:
            # Generate sample data
            if generate_sample_data(output_path, num_samples=100):
                if verify_files:
                    if verify_jsonl_file(output_path):
                        success_count += 1
                    else:
                        output_path.unlink()
                else:
                    success_count += 1
        else:
            # Download from Hugging Face
            if download_and_save_dataset(
                config["dataset_name"],
                output_path,
                config["description"]
            ):
                if verify_files:
                    if verify_jsonl_file(output_path):
                        success_count += 1
                    else:
                        output_path.unlink()
                else:
                    success_count += 1
    
    if success_count == total_datasets:
        console.print(f"[green]✓ Successfully {'generated' if use_sample_data else 'downloaded'} all {total_datasets} datasets[/green]")
        return True
    else:
        console.print(f"[red]✗ Only {success_count}/{total_datasets} datasets {'generated' if use_sample_data else 'downloaded'} successfully[/red]")
        return False


def list_available_datasets() -> None:
    """List available datasets and their information."""
    console.print("[bold]Available AgentNet datasets:[/bold]")
    for key, config in DATASET_CONFIGS.items():
        console.print(f"  [bold]{key}[/bold]: {config['description']}")
        console.print(f"    Dataset: {config['dataset_name']}")
        console.print(f"    Output: {config['output_filename']}")
        console.print()


def main():
    """Main function to handle command line arguments and download datasets."""
    parser = argparse.ArgumentParser(
        description="Download AgentNet datasets using Hugging Face datasets library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_agentnet.py
  python scripts/download_agentnet.py --data-dir ./data
  python scripts/download_agentnet.py --datasets agentnet_clicks
  python scripts/download_agentnet.py --sample-data
  python scripts/download_agentnet.py --no-verify
  python scripts/download_agentnet.py --list-datasets
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save dataset files (default: ./data)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        help="Specific datasets to download (default: all)"
    )
    
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Generate sample data instead of downloading from Hugging Face"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of downloaded files"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_available_datasets()
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
    
    success = download_datasets(
        data_dir=args.data_dir,
        datasets_to_download=args.datasets,
        verify_files=not args.no_verify,
        use_sample_data=args.sample_data
    )
    
    if success:
        console.print(f"\n[bold green]Datasets ready![/bold green]")
        console.print(f"Files saved to: {args.data_dir.absolute()}")
        console.print("\nYou can now run the analysis script:")
        console.print(f"python scripts/agentnet.py")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]Download failed![/bold red]")
        console.print("Try using --sample-data flag to generate test data, or check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()