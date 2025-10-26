"""Utilities for downloading and summarising the agentsea/wave-ui dataset.

This script mirrors the rich CLI experience delivered by `agentnet.py`,
tailored to the Wave UI image grounding corpus hosted on Hugging Face.

Key capabilities:
  * optional snapshot download of the parquet shards and metadata
  * rich console summaries of split sizes and categorical distributions
  * lightweight schema visualisation
  * sampling helpers to inspect representative instructions
  * verification helper that compares the observed pair count to an
    expected target (80k by default)

Example usage (remote analysis only):

```bash
python3 scripts/wave_ui.py
```

Download the shards locally before analysing them:

```bash
python3 scripts/wave_ui.py --download-dir ./data/wave-ui --use-downloaded
```
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, snapshot_download
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()

DEFAULT_REPO_ID = "agentsea/wave-ui"
DEFAULT_SPLITS: Tuple[str, ...] = ("train", "validation", "test")


@dataclass
class SplitStats:
    """Aggregated summary for a dataset split."""

    name: str
    row_count: int
    platform_counts: Counter[str] = field(default_factory=Counter)
    type_counts: Counter[str] = field(default_factory=Counter)
    language_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    website_counts: Counter[str] = field(default_factory=Counter)
    action_counts: Counter[str] = field(default_factory=Counter)
    unique_sources: int = 0
    avg_resolution: Optional[Tuple[float, float]] = None
    samples: List[dict[str, Any]] = field(default_factory=list)


@dataclass
class WaveUIReport:
    """Container for the combined dataset report."""

    repo_id: str
    splits: List[SplitStats]
    schema_fields: List[Tuple[str, str]]

    @property
    def total_rows(self) -> int:
        return sum(split.row_count for split in self.splits)


def _preview(value: Optional[str], limit: int = 120) -> Optional[str]:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


DOMAIN_REGEX = re.compile(
    r"(?:https?://|www\.)?([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)", re.IGNORECASE
)


def _extract_domains_from_row(row: dict[str, Any]) -> Iterable[str]:
    domains: set[str] = set()
    for key in ("instruction", "description", "purpose", "expectation", "name"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        for match in DOMAIN_REGEX.finditer(value):
            domain = match.group(1).lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domains.add(domain)
    return domains


def _normalise_action(instruction: Any, element_type: Any) -> Optional[str]:
    if isinstance(instruction, str):
        text = instruction.strip()
        if text:
            if "->" in text:
                candidate = text.split("->")[-1].strip()
                if candidate:
                    return candidate.upper().replace(" ", "_")
            if text.startswith("[") and "]" in text:
                bracket = text.split("]", 1)[0]
                label = bracket[1:].strip()
                if label:
                    return label.upper().replace(" ", "_")
            if "," in text:
                parts = [
                    segment.strip() for segment in text.split(",") if segment.strip()
                ]
                if parts:
                    return parts[-1].upper().replace(" ", "_")
            first_word = text.split()[0]
            if first_word:
                return first_word.upper().replace(" ", "_")
    if isinstance(element_type, str):
        label = element_type.strip()
        if label:
            return label.upper().replace(" ", "_")
    return None


def _coerce_local_data_root(path: Path) -> Path:
    """Resolve the directory that actually contains the parquet shards."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local path '{path}' does not exist")
    if path.is_file():
        raise FileNotFoundError(
            f"Expected a directory containing parquet files, got file '{path}'"
        )

    # Accept the directory as-is if the shards live directly inside.
    if any(path.glob("train-*.parquet")):
        return path

    candidate = path / "data"
    if candidate.is_dir() and any(candidate.glob("train-*.parquet")):
        return candidate

    raise FileNotFoundError(
        f"Could not locate parquet shards under '{path}'. "
        "Place the dataset shards directly in this directory or inside a 'data/' subdir."
    )


def _resolve_split_paths(
    split: str,
    repo_id: str,
    fs: HfFileSystem,
    data_root: str,
) -> List[str]:
    pattern = f"{data_root}/{split}-*.parquet"
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No parquet shards found for split '{split}' at '{pattern}'"
        )
    return paths


def _count_rows_with_metadata(
    dataset_paths: Sequence[str],
    filesystem: pafs.FileSystem,
) -> int:
    total_rows = 0
    for path in dataset_paths:
        try:
            metadata = pq.read_metadata(path, filesystem=filesystem)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to read parquet metadata for '{path}': {exc}"
            ) from exc
        total_rows += metadata.num_rows
    return int(total_rows)


def _collect_split_stats(
    split: str,
    dataset_paths: Sequence[str],
    filesystem: pafs.FileSystem,
    sample_size: int,
    batch_size: int,
    scan_stats: bool,
) -> Tuple[SplitStats, List[Tuple[str, str]]]:
    dataset = ds.dataset(dataset_paths, filesystem=filesystem, format="parquet")

    schema_fields = [(field.name, str(field.type)) for field in dataset.schema]
    row_count = _count_rows_with_metadata(dataset_paths, filesystem)

    platform_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    website_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    unique_sources_set: set[str] = set()

    total_width = 0.0
    total_height = 0.0
    resolution_count = 0

    if scan_stats:
        columns_needed = {
            "platform",
            "type",
            "language",
            "source",
            "resolution",
            "instruction",
            "description",
            "purpose",
            "expectation",
            "name",
        }
        columns_to_scan = [col for col in columns_needed if col in dataset.schema.names]

        if columns_to_scan:
            try:
                for batch in dataset.to_batches(
                    columns=columns_to_scan, batch_size=batch_size, use_threads=False
                ):
                    if "platform" in batch.schema.names:
                        platform_counts.update(
                            value
                            for value in batch.column("platform").to_pylist()
                            if isinstance(value, str) and value
                        )
                    if "type" in batch.schema.names:
                        type_counts.update(
                            value
                            for value in batch.column("type").to_pylist()
                            if isinstance(value, str) and value
                        )
                    if "language" in batch.schema.names:
                        language_counts.update(
                            value
                            for value in batch.column("language").to_pylist()
                            if isinstance(value, str) and value
                        )
                    if "source" in batch.schema.names:
                        values = [
                            value
                            for value in batch.column("source").to_pylist()
                            if isinstance(value, str) and value
                        ]
                        source_counts.update(values)
                        unique_sources_set.update(values)
                    if "resolution" in batch.schema.names:
                        for entry in batch.column("resolution").to_pylist():
                            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                                width, height = entry[0], entry[1]
                                if isinstance(width, (int, float)) and isinstance(
                                    height, (int, float)
                                ):
                                    total_width += float(width)
                                    total_height += float(height)
                                    resolution_count += 1

                    row_level_columns = {
                        "instruction",
                        "description",
                        "purpose",
                        "expectation",
                        "name",
                        "type",
                    }
                    if any(
                        column in batch.schema.names for column in row_level_columns
                    ):
                        for row in batch.to_pylist():
                            for domain in _extract_domains_from_row(row):
                                website_counts[domain] += 1
                            action_label = _normalise_action(
                                row.get("instruction"), row.get("type")
                            )
                            if action_label:
                                action_counts[action_label] += 1
            except Exception as exc:
                console.print(
                    Text.assemble(
                        Text("Warning: ", style="yellow"),
                        Text(
                            f"failed to stream detailed statistics for split '{split}' due to: {exc}. Continuing with metadata-only summary."
                        ),
                    )
                )
                platform_counts.clear()
                type_counts.clear()
                language_counts.clear()
                source_counts.clear()
                website_counts.clear()
                action_counts.clear()
                unique_sources_set.clear()
                total_width = 0.0
                total_height = 0.0
                resolution_count = 0

    avg_resolution: Optional[Tuple[float, float]] = None
    if resolution_count:
        avg_resolution = (
            round(total_width / resolution_count, 2),
            round(total_height / resolution_count, 2),
        )

    sample_columns = [
        col
        for col in (
            "instruction",
            "platform",
            "type",
            "name",
            "description",
            "source",
            "language",
            "bbox",
            "resolution",
            "purpose",
            "expectation",
        )
        if col in dataset.schema.names
    ]

    samples: List[dict[str, Any]] = []
    if sample_size > 0 and sample_columns:
        try:
            table = dataset.head(sample_size, columns=sample_columns, use_threads=False)
        except Exception as exc:
            console.print(
                Text.assemble(
                    Text("Warning: ", style="yellow"),
                    Text(
                        f"could not collect samples for split '{split}' due to: {exc}."
                    ),
                )
            )
            table = None

        if table is not None:
            for row in table.to_pylist():
                action_label = _normalise_action(
                    row.get("instruction"), row.get("type")
                )
                if not scan_stats:
                    source_value = row.get("source")
                    if isinstance(source_value, str) and source_value:
                        source_counts.update([source_value])
                        unique_sources_set.add(source_value)
                    for domain in _extract_domains_from_row(row):
                        website_counts[domain] += 1
                    if action_label:
                        action_counts.update([action_label])
                normalised: dict[str, Any] = {}
                for key, value in row.items():
                    if isinstance(value, str):
                        normalised[key] = _preview(value)
                    elif key == "bbox" and isinstance(value, list):
                        normalised[key] = [round(float(v), 2) for v in value]
                    else:
                        normalised[key] = value
                if action_label:
                    normalised.setdefault("action", action_label)
                samples.append(normalised)

            if not avg_resolution and samples:
                widths: List[float] = []
                heights: List[float] = []
                for sample in samples:
                    resolution = sample.get("resolution")
                    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
                        width, height = resolution[0], resolution[1]
                        if isinstance(width, (int, float)) and isinstance(
                            height, (int, float)
                        ):
                            widths.append(float(width))
                            heights.append(float(height))
                if widths and heights:
                    avg_resolution = (
                        round(sum(widths) / len(widths), 2),
                        round(sum(heights) / len(heights), 2),
                    )

    stats = SplitStats(
        name=split,
        row_count=row_count,
        platform_counts=platform_counts,
        type_counts=type_counts,
        language_counts=language_counts,
        source_counts=source_counts,
        website_counts=website_counts,
        action_counts=action_counts,
        unique_sources=len(unique_sources_set)
        if unique_sources_set
        else len(source_counts),
        avg_resolution=avg_resolution,
        samples=samples,
    )

    del dataset  # ensure background readers are cleaned up before returning
    return stats, schema_fields


def collect_wave_ui_stats(
    repo_id: str = DEFAULT_REPO_ID,
    splits: Sequence[str] = DEFAULT_SPLITS,
    local_data_root: Optional[Path] = None,
    sample_size: int = 5,
    batch_size: int = 1024,
    scan_stats: bool = False,
) -> WaveUIReport:
    """Gather summary statistics for the requested dataset splits."""

    split_stats: List[SplitStats] = []
    schema_fields: List[Tuple[str, str]] = []

    if local_data_root is None:
        filesystem: pafs.FileSystem = HfFileSystem()
        data_root = f"datasets/{repo_id}/data"
        for index, split in enumerate(splits):
            paths = _resolve_split_paths(split, repo_id, filesystem, data_root)
            stats, schema = _collect_split_stats(
                split=split,
                dataset_paths=paths,
                filesystem=filesystem,
                sample_size=sample_size,
                batch_size=batch_size,
                scan_stats=scan_stats,
            )
            split_stats.append(stats)
            if index == 0:
                schema_fields = schema
    else:
        data_root = _coerce_local_data_root(local_data_root)
        filesystem = pafs.LocalFileSystem()
        for index, split in enumerate(splits):
            pattern = sorted(str(path) for path in data_root.glob(f"{split}-*.parquet"))
            if not pattern:
                raise FileNotFoundError(
                    f"No parquet shards found for split '{split}' under '{data_root}'"
                )
            stats, schema = _collect_split_stats(
                split=split,
                dataset_paths=pattern,
                filesystem=filesystem,
                sample_size=sample_size,
                batch_size=batch_size,
                scan_stats=scan_stats,
            )
            split_stats.append(stats)
            if index == 0:
                schema_fields = schema

    return WaveUIReport(
        repo_id=repo_id, splits=split_stats, schema_fields=schema_fields
    )


def download_wave_ui_dataset(
    target_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
) -> Path:
    """Download the dataset snapshot into ``target_dir`` and return its path."""

    target_dir = target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Text.assemble(
            Text("Downloading ", style="bold"),
            Text(repo_id, style="cyan"),
            Text(f" → {target_dir}"),
        )
    )

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns or ("data/*.parquet", "README.md", "LICENSE.md"),
    )

    resolved = Path(snapshot_path).expanduser().resolve()
    console.print(Text(f"Snapshot ready at {resolved}", style="green"))
    return resolved


def _render_schema(schema_fields: Sequence[Tuple[str, str]]) -> None:
    tree = Tree(Text("Schema", style="bold magenta"))
    for name, dtype in schema_fields:
        tree.add(
            Text.assemble(Text(name, style="cyan"), Text(f": {dtype}", style="dim"))
        )
    console.print(tree)


def _render_counter_table(title: str, counter: Counter[str], limit: int = 10) -> None:
    if not counter:
        return
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("value", style="bold")
    table.add_column("count", justify="right")
    for key, value in counter.most_common(limit):
        table.add_row(key, f"{value:,}")
    console.print(table)


def _render_samples(split: SplitStats) -> None:
    if not split.samples:
        return

    table = Table(title=f"[dim]{split.name} samples", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right")
    table.add_column("instruction", style="bold")
    table.add_column("action", justify="left")
    table.add_column("type")
    table.add_column("platform")
    table.add_column("bbox")
    table.add_column("source")

    for index, sample in enumerate(split.samples, start=1):
        bbox = sample.get("bbox")
        bbox_display = "" if bbox is None else str(bbox)
        table.add_row(
            str(index),
            sample.get("instruction") or "",
            sample.get("action") or "",
            sample.get("type") or "",
            sample.get("platform") or "",
            bbox_display,
            sample.get("source") or "",
        )

    console.print(table)


def render_report(report: WaveUIReport, expected_pairs: int = 80_000) -> None:
    console.print(Text(f"Wave UI Dataset · {report.repo_id}", style="bold white"))

    overall = Table(title="[dim]Rows per split", box=box.SIMPLE_HEAVY, show_lines=False)
    overall.add_column("split", style="bold")
    overall.add_column("rows", justify="right")
    overall.add_column("avg resolution", justify="center")
    overall.add_column("unique sources", justify="right")

    for split in report.splits:
        resolution = (
            f"{int(split.avg_resolution[0])}×{int(split.avg_resolution[1])}"
            if split.avg_resolution
            else "–"
        )
        overall.add_row(
            split.name,
            f"{split.row_count:,}",
            resolution,
            f"{split.unique_sources:,}" if split.unique_sources else "–",
        )

    console.print(overall)

    total_rows = report.total_rows
    delta = total_rows - expected_pairs
    delta_text = f"Δ {delta:+,} vs expected"
    style = "green" if delta == 0 else ("yellow" if abs(delta) <= 1_000 else "red")
    verification = Text.assemble(
        Text("Total image+annotation pairs: ", style="bold"),
        Text(f"{total_rows:,}", style="cyan"),
        Text("  ("),
        Text(delta_text, style=style),
        Text(")"),
    )
    console.print(verification)

    _render_schema(report.schema_fields)

    for split in report.splits:
        console.print(Text(f"Split: {split.name}", style="bold cyan"))

        _render_counter_table(
            f"[dim]{split.name} · top platforms", split.platform_counts
        )
        _render_counter_table(
            f"[dim]{split.name} · top element types", split.type_counts
        )
        _render_counter_table(f"[dim]{split.name} · top actions", split.action_counts)
        _render_counter_table(f"[dim]{split.name} · top sources", split.source_counts)
        _render_counter_table(f"[dim]{split.name} · top websites", split.website_counts)
        _render_counter_table(
            f"[dim]{split.name} · top languages", split.language_counts
        )

        _render_samples(split)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and analyse the agentsea/wave-ui dataset",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Dataset repository id on Hugging Face",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to inspect (default: train validation test)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Optional directory to store a dataset snapshot before analysis",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Optional dataset revision (branch, tag, or commit sha)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the dataset snapshot and exit without running analysis",
    )
    parser.add_argument(
        "--use-downloaded",
        action="store_true",
        help="Analyse the freshly downloaded snapshot instead of remote parquet files",
    )
    parser.add_argument(
        "--local-data",
        type=Path,
        help="Path to pre-downloaded dataset shards (directory containing parquet files or a data/ directory)",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=80_000,
        help="Expected number of image+annotation pairs for verification",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of example rows to display per split",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size used while streaming parquet statistics",
    )
    parser.add_argument(
        "--full-stats",
        action="store_true",
        help="Scan the entire dataset to compute categorical distributions (downloads several GB when remote).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.download_dir is None and args.download_only:
        parser.error("--download-only requires --download-dir")

    downloaded_path: Optional[Path] = None
    if args.download_dir is not None:
        downloaded_path = download_wave_ui_dataset(
            target_dir=args.download_dir,
            repo_id=args.repo_id,
            revision=args.revision,
        )
        if args.download_only:
            return

    analysis_root: Optional[Path] = None
    if args.local_data is not None:
        analysis_root = args.local_data
    if args.use_downloaded and downloaded_path is not None:
        analysis_root = downloaded_path

    report = collect_wave_ui_stats(
        repo_id=args.repo_id,
        splits=tuple(args.splits),
        local_data_root=analysis_root,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        scan_stats=args.full_stats,
    )
    render_report(report, expected_pairs=args.expected_pairs)


if __name__ == "__main__":
    main()
