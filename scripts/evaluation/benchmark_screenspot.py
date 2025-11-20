#!/usr/bin/env python3
"""
Benchmark model on ScreenSpot-Pro dataset.
Downloads from HuggingFace, transforms to ShareGPT format, and runs inference.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import shutil

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image as PILImage
from peft import PeftModel


def load_model(model_name_or_path: str, adapter_path: str = None, device: str = "cuda"):
    """Load model and processor for inference."""
    print(f"Loading model from {model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, device_map=device)
    
    model.eval()
    return model, processor


def download_screenspot_pro(output_dir: str = "screenspot_pro"):
    """Download ScreenSpot-Pro dataset from HuggingFace and transform to ShareGPT format."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print("Downloading ScreenSpot-Pro from HuggingFace...")
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("likaixin/ScreenSpot-Pro")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    records = []
    system_prompt = "You are a GUI automation assistant. Given an image and a user instruction, output the exact pyautogui.click(x, y) command to execute the action. Coordinates are normalized to 1400x800 resolution."
    
    print("Transforming dataset to ShareGPT format...")
    # Use the train split (or default split)
    dataset_split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    
    for i, sample in enumerate(tqdm(dataset_split, desc="Processing samples")):
        # Get image from dataset
        image = sample.get("image")
        if image is None:
            continue
        
        # Save image to output directory
        image_filename = f"{i:06d}.png"
        dest_image_path = images_dir / image_filename
        image.save(dest_image_path)
        
        # Extract instruction/prompt from various possible fields
        instruction = None
        # Try sample fields
        for field in ["instruction", "text", "prompt", "query", "task", "label"]:
            if field in sample and sample[field] is not None:
                value = sample[field]
                if value:
                    if field == "label":
                        # If it's a label, create an instruction from it
                        instruction = f"identify elements in this {value} interface"
                    else:
                        instruction = str(value)
                    break
        
        if not instruction:
            instruction = "click on the target element"
        
        # Create ShareGPT format record
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<image>\n{instruction}"}
            ],
            "images": [f"images/{image_filename}"],
            "instruction": instruction,
            "sample_id": i
        }
        
        # Store label if available
        if "label" in sample and sample["label"] is not None:
            record["gt_label"] = sample["label"]
        
        # Store ground truth bbox if available
        for bbox_field in ["bbox", "bounding_box", "gt_bbox"]:
            if bbox_field in sample and sample[bbox_field] is not None:
                record["gt_bbox"] = sample[bbox_field]
                break
        
        records.append(record)
    
    # Save transformed dataset
    jsonl_path = output_path / "data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Transformed {len(records)} samples to {jsonl_path}")
    return records, str(output_path)


def run_inference(model, processor, records: List[Dict], media_dir: str, device: str, output_file: str):
    """Run inference on records and save predictions."""
    print(f"Running inference on {len(records)} samples...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(records, desc="Inference"):
            messages = record.get("messages", [])
            images = record.get("images", [])
            
            if not messages or not images:
                failed += 1
                continue
            
            image_path = images[0]
            full_image_path = os.path.join(media_dir, image_path)
            
            if not os.path.exists(full_image_path):
                failed += 1
                continue
            
            try:
                image = PILImage.open(full_image_path).convert("RGB")
            except Exception as e:
                failed += 1
                continue
            
            try:
                inputs = processor(messages=messages, images=image, return_tensors="pt").to(device)
            except Exception:
                try:
                    inputs = processor(text=messages, images=image, return_tensors="pt").to(device)
                except Exception as e:
                    failed += 1
                    continue
            
            with torch.no_grad():
                try:
                    input_length = inputs.input_ids.shape[1]
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                    )
                    generated_ids_trimmed = generated_ids[0][input_length:]
                    output_text = processor.decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ).strip()
                    
                    result = {
                        "input": {
                            "prompt": messages[1]["content"] if len(messages) > 1 else "",
                            "image_path": image_path,
                        },
                        "output": output_text,
                        "sample_id": record.get("sample_id"),
                        "instruction": record.get("instruction"),
                    }
                    
                    if "gt_bbox" in record:
                        result["gt_bbox"] = record["gt_bbox"]
                        result["gt_label"] = record.get("gt_label")
                    
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    successful += 1
                except Exception as e:
                    failed += 1
                    continue
    
    print(f"\n✓ Inference complete: {successful} successful, {failed} failed")
    print(f"Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model on ScreenSpot-Pro dataset")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--output-dir", type=str, default="screenspot_pro", help="Output directory for dataset")
    parser.add_argument("--predictions", type=str, default="screenspot_pro_predictions.jsonl", help="Output predictions file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if dataset already exists")
    
    args = parser.parse_args()
    
    # Download and transform dataset
    data_jsonl = Path(args.output_dir) / "data.jsonl"
    if args.skip_download and data_jsonl.exists():
        print(f"Loading existing dataset from {data_jsonl}...")
        records = []
        with open(data_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        media_dir = args.output_dir
    else:
        records, media_dir = download_screenspot_pro(args.output_dir)
    
    if not records:
        print("Error: No records found!")
        return 1
    
    # Load model
    model, processor = load_model(args.model_name_or_path, args.adapter_path, args.device)
    
    # Run inference
    run_inference(model, processor, records, media_dir, args.device, args.predictions)
    
    print(f"\n✓ Benchmark complete! Predictions saved to: {args.predictions}")
    return 0


if __name__ == "__main__":
    exit(main())

