#!/usr/bin/env python3
"""
Calculate accuracy metrics for the trained model on test/validation data.
Runs inference and evaluates grounding performance.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import sys

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image as PILImage
from peft import PeftModel

# Import evaluation function
sys.path.insert(0, str(Path(__file__).parent))
from grounding_eval import evaluate_grounding, print_results, parse_pyautogui_click


def load_model(model_path: str = None, adapter_path: str = None, base_model: str = "Qwen/Qwen3-VL-8B-Instruct"):
    """Load model for inference."""
    if model_path:
        print(f"Loading merged model: {model_path}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif adapter_path:
        print(f"Loading base model: {base_model}")
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, device_map="auto")
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        raise ValueError("Either --model-path or --adapter-path must be provided")
    
    model.eval()
    return model, processor


def load_test_data(data_path: Path) -> Dict:
    """Load test data in ShareGPT format."""
    ground_truth = {}
    
    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract image path and ground truth
                conversations = data.get("conversations", [])
                if not conversations:
                    continue
                
                # Get image path
                image_path = None
                for msg in conversations:
                    if msg.get("role") == "user":
                        images = msg.get("images", [])
                        if images:
                            image_path = images[0]
                            break
                
                if not image_path:
                    continue
                
                # Get ground truth from assistant response
                gt_text = None
                for msg in conversations:
                    if msg.get("role") == "assistant":
                        gt_text = msg.get("content")
                        break
                
                if not gt_text:
                    continue
                
                # Parse ground truth coordinates
                gt_coords = parse_pyautogui_click(gt_text)
                if not gt_coords:
                    print(f"Warning: Could not parse ground truth at line {line_num}: {gt_text[:100]}")
                    continue
                
                # Extract bbox from metadata if available
                bbox = data.get("bbox")
                resolution = data.get("resolution", [1400, 800])
                sample_id = data.get("id", f"sample_{line_num}")
                
                # If no explicit bbox, create one from coordinates (assuming small target area)
                if not bbox:
                    x, y = gt_coords
                    # Create a small bbox around the click point (20px radius)
                    bbox = [x - 20, y - 20, x + 20, y + 20]
                
                ground_truth[image_path] = {
                    "id": sample_id,
                    "bbox": bbox,
                    "resolution": resolution,
                    "ground_truth_text": gt_text,
                    "ground_truth_coords": gt_coords
                }
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    return ground_truth


def run_inference(model, processor, test_data: Dict, data_dir: Path, max_samples: int = None) -> Dict[str, str]:
    """Run inference on test data."""
    predictions = {}
    samples = list(test_data.items())
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"\nRunning inference on {len(samples)} samples...")
    
    for image_path, gt_data in tqdm(samples, desc="Inference"):
        try:
            # Load image
            full_image_path = data_dir / image_path
            if not full_image_path.exists():
                print(f"Warning: Image not found: {full_image_path}")
                continue
            
            image = PILImage.open(full_image_path).convert("RGB")
            
            # Get instruction from ground truth text (extract the instruction before "Output:")
            gt_text = gt_data["ground_truth_text"]
            if "Output:" in gt_text:
                instruction = gt_text.split("Output:")[0].strip()
            else:
                # Fallback: use a generic instruction
                instruction = "Click on the target element as instructed."
            
            # Prepare prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a GUI automation assistant. Given an image and a user instruction, output the exact pyautogui.click(x, y) command to execute the action. Coordinates are normalized to 1400x800 resolution."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction}
                    ]
                }
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Process inputs
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # Decode output
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            prediction = processor.decode(generated_ids, skip_special_tokens=True)
            
            predictions[image_path] = prediction
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy metrics for grounding model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to test/val JSONL file")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing images")
    parser.add_argument("--model-path", type=str, help="Path to merged model")
    parser.add_argument("--adapter-path", type=str, help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Base model name")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--output", type=str, help="Output file for predictions (JSONL)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, processor = load_model(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        base_model=args.base_model
    )
    
    # Load test data
    print(f"Loading test data from {args.data_path}...")
    test_data = load_test_data(Path(args.data_path))
    print(f"Loaded {len(test_data)} test samples")
    
    # Run inference
    predictions = run_inference(
        model,
        processor,
        test_data,
        Path(args.data_dir),
        max_samples=args.max_samples
    )
    
    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for image_path, prediction in predictions.items():
                record = {
                    "image_path": image_path,
                    "prediction": prediction,
                    **test_data[image_path]
                }
                f.write(json.dumps(record) + '\n')
        print(f"\n✓ Saved predictions to: {output_path}")
    
    # Evaluate
    print("\nEvaluating predictions...")
    results = evaluate_grounding(test_data, predictions)
    
    # Print results
    print_results(results, verbose=args.verbose)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct']} / {results['total']}")
    if results.get('distance_stats'):
        stats = results['distance_stats']
        print(f"Mean Distance to Target: {stats['mean']:.2f} pixels")
        print(f"Within 10px: {stats['within_10px_percentage']:.2f}%")
        print(f"Within 20px: {stats['within_20px_percentage']:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()


