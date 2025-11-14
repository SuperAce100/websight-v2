#!/usr/bin/env python3
"""
Run the ScreenSpot-Pro benchmark locally using the WebSight v2 grounded model.

The script expects the dataset (images + data.jsonl) to already exist on disk.
It simply loads that cache, runs inference, and writes predictions to JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from screenspot_pro_utils import ensure_dataset


def timestamped_predictions_path(base_dir: str | os.PathLike[str]) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"screenspot_pro_{ts}.jsonl"


def load_model(
    model_name_or_path: str,
    adapter_path: Optional[str],
    device: str,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    print(f"Loading processor + model from {model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        from peft import PeftModel  # Lazy import to avoid dependency unless needed

        model = PeftModel.from_pretrained(model, adapter_path, device_map=device)

    model.eval()
    return model, processor


def render_messages(processor, messages: List[Dict]) -> str:
    """
    Convert structured messages into a single prompt string compatible with processors
    that do not accept the `messages` kwarg.
    """
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    rendered: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            rendered.append(f"{role}: {content}")
        elif isinstance(content, list):
            # Some processors (OpenAI-style) use list[{type,text}]
            text_chunks = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_chunks.append(str(part.get("text", "")))
            rendered.append(f"{role}: {' '.join(text_chunks)}")
    return "\n".join(rendered)


def _iter_records(records: List[Dict], limit: Optional[int]) -> List[Dict]:
    if limit is None:
        return records
    return records[:limit]


def run_inference(
    model,
    processor,
    records: List[Dict],
    media_dir: str,
    device: str,
    output_file: Path,
    max_new_tokens: int = 320,
) -> Tuple[int, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0

    # Ensure file starts empty before appending per-sample entries
    output_file.write_text("", encoding="utf-8")

    for record in tqdm(records, desc="ScreenSpot-Pro inference"):
        messages = record.get("messages", [])
        image_rel = record.get("image_path") or (
            record.get("images", [""])[0] if record.get("images") else None
        )

        if not messages or not image_rel:
            failures += 1
            continue

        full_image_path = os.path.join(media_dir, image_rel)
        if not os.path.exists(full_image_path):
            failures += 1
            continue

        try:
            image = Image.open(full_image_path).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Failed to open {full_image_path}: {exc}")
            failures += 1
            continue

        try:
            rendered_prompt = render_messages(processor, messages)
            inputs = processor(
                text=rendered_prompt,
                images=image,
                return_tensors="pt",
            ).to(device)
        except Exception as exc:
            print(f"[WARN] Failed to tokenize sample {record.get('sample_id')}: {exc}")
            failures += 1
            continue

        with torch.no_grad():
            try:
                input_length = inputs.input_ids.shape[1]
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                generated_ids_trimmed = generated_ids[0][input_length:]
                output_text = processor.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            except Exception as exc:
                print(f"[WARN] Generation failed: {exc}")
                failures += 1
                continue

        payload = {
            "input": {
                "prompt": messages[-1]["content"] if messages else "",
                "image_path": image_rel,
            },
            "output": output_text,
            "sample_id": record.get("sample_id"),
            "instruction": record.get("instruction"),
        }

        for key in ("bbox", "gt_bbox", "gt_label"):
            if key in record:
                payload[key] = record[key]

        with output_file.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
        successes += 1

    return successes, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ScreenSpot-Pro benchmark against a HuggingFace model."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Asanshay/websight-v2-grounded",
        help="HuggingFace model id or local path (default: Asanshay/websight-v2-grounded)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="screenspot_pro",
        help="Directory containing processed ScreenSpot-Pro data.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Output JSONL path. If omitted, a timestamped file in runs/screenspot_pro/ is used.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="runs/screenspot_pro",
        help="Directory used when --predictions is not provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device or device map for HF `device_map` (default: cuda).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run inference on only the first N records (after dataset is loaded).",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit number of samples loaded from the dataset.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=320,
        help="Generation max_new_tokens (default: 320).",
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=None,
        help="Randomly select this many samples before inference.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="Seed for random subset selection (default: random module default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records, media_dir = ensure_dataset(
        output_dir=args.output_dir,
        limit=args.dataset_limit,
    )

    if args.subset_count is not None:
        if args.subset_count > len(records):
            raise ValueError(
                f"subset-count ({args.subset_count}) exceeds dataset size ({len(records)})"
            )
        rng = random.Random(args.subset_seed)
        chosen_indices = sorted(rng.sample(range(len(records)), args.subset_count))
        records = [records[i] for i in chosen_indices]
        print(
            f"Random subset: kept {len(records)} samples "
            f"(seed={args.subset_seed if args.subset_seed is not None else 'system'})"
        )

    if args.limit:
        records = _iter_records(records, args.limit)
        print(f"Limiting inference to first {len(records)} samples.")

    predictions_path = (
        Path(args.predictions).expanduser()
        if args.predictions
        else timestamped_predictions_path(args.predictions_dir)
    )

    model, processor = load_model(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        device=args.device,
    )

    successes, failures = run_inference(
        model=model,
        processor=processor,
        records=records,
        media_dir=media_dir,
        device=args.device,
        output_file=predictions_path,
        max_new_tokens=args.max_new_tokens,
    )

    total = successes + failures
    accuracy = successes / total if total else 0.0
    print(
        f"\n✓ Benchmark finished. Success: {successes} | Failures: {failures} | "
        f"Complete: {accuracy:.2%}"
    )
    print(f"Predictions saved to: {predictions_path}")
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
