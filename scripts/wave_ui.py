"""Minimal Wave UI dataset helper."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.dataset as ds
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "agentsea/wave-ui"
BATCH_SIZE = 1024
_DROP = object()


def cleanse(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _DROP
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            filtered = cleanse(item)
            if filtered is _DROP:
                continue
            cleaned[key] = filtered
        return cleaned
    if isinstance(value, list):
        cleaned_list: list[Any] = []
        for item in value:
            filtered = cleanse(item)
            if filtered is _DROP:
                continue
            cleaned_list.append(filtered)
        return cleaned_list
        return value


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix == ".jsonl":
        yield root
        return
    jsonl_path = root / "data.jsonl"
    if jsonl_path.exists():
        yield jsonl_path
        return
    for path in sorted(root.glob("*.jsonl")):
        yield path
    for path in sorted(root.glob("**/*.jsonl")):
        if path.parent != root:
            yield path


    parser = argparse.ArgumentParser(
    description="Download, extract, and summarise the Wave UI dataset.",
    )
    parser.add_argument(
    "data_dir",
        type=Path,
    help="Directory to store or read the extracted dataset",
    )
parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
    "--download",
        action="store_true",
    help="Download parquet shards and extract them into JSONL+images",
    )
    parser.add_argument(
    "--limit",
        type=int,
    help="Optional limit when summarising existing JSONL data",
)
args = parser.parse_args()

base_dir = args.data_dir.expanduser().resolve()
base_dir.mkdir(parents=True, exist_ok=True)

counts = Counter()
row_count = 0
jsonl_output: Path | None = None
images_dir: Path | None = None

if args.download:
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=raw_dir,
            local_dir_use_symlinks=False,
            allow_patterns=("data/*.parquet", "README.md", "LICENSE.md"),
        )
    ).expanduser().resolve()

    parquet_root = snapshot_path / "data"
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output = base_dir / "data.jsonl"

    with jsonl_output.open("w", encoding="utf-8") as jsonl_file:
        image_index = 0
        for parquet_path in sorted(parquet_root.glob("*.parquet")):
            dataset = ds.dataset(str(parquet_path), format="parquet")
            split_name = parquet_path.stem.split("-")[0]
            for batch in dataset.to_batches(batch_size=BATCH_SIZE, use_threads=True):
                for row in batch.to_pylist():
                    image_rel = None
                    image_entry = row.get("image")
                    if isinstance(image_entry, dict):
                        image_bytes = image_entry.get("bytes")
                        hint = image_entry.get("path") or ""
                        suffix = Path(hint).suffix or ".png"
                        filename = f"{image_index:06d}{suffix}"
                        if image_bytes:
                            (images_dir / filename).write_bytes(image_bytes)
                            image_rel = f"images/{filename}"
                        image_index += 1

                    record: dict[str, Any] = {}
                    for key, value in row.items():
                        if key == "image":
                            continue
                        cleaned = cleanse(value)
                        if cleaned is _DROP:
                            continue
                        record[key] = cleaned

                    if image_rel is not None:
                        record["image_path"] = image_rel
                    record["split"] = split_name

                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                    label = record.get("type")
                    if isinstance(label, str) and label:
                        counts[label] += 1
                    row_count += 1
else:
    files = list(iter_jsonl_files(base_dir))
    limit = args.limit
    for file in files:
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and row_count >= limit:
                    break
                record = json.loads(line)
                label = record.get("type")
                if isinstance(label, str) and label:
                    counts[label] += 1
                row_count += 1
        if limit is not None and row_count >= limit:
            break

print(f"Rows processed: {row_count}")
if jsonl_output is not None:
    print(f"JSONL written to: {jsonl_output}")
if images_dir is not None:
    print(f"Images directory: {images_dir}")

for label, count in counts.most_common():
    print(f"{label}\t{count}")
