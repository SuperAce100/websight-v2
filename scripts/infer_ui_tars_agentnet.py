#!/usr/bin/env python3
"""
Run a single inference turn on the merged UI-TARS AgentNet model.

Defaults match the cluster layout:
- Model: /hai/scratch/asanshay/websight-v2/merged/ui-tars-1.5-7b-agentnet-full
- Image: /hai/users/a/s/asanshay/websight-v2/sample.png
- Prompt: "Click the Add New button"
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-shot inference for UI-TARS AgentNet"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/hai/users/a/s/asanshay/websight-v2/sample.png",
        help="Path to the screenshot image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Click the Add New button",
        help="User instruction for the model.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Asanshay/websight-2-7B-kto",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Torch dtype for loading the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")

    system_prompt = """
You are a GUI agent. You are given a task and a screenshot of the screen. You
need to perform a series of pyautogui actions to complete the task.

For each step, provide your response in this format:


**Code:** Finally, output the action as a PyAutoGUI command (e.g.
pyautogui.click(x, y), etc.)


**Thought:**
- Next Action Analysis:
  - List possible next actions based on current state
  - Evaluate options considering current state and previous actions
  - Propose most logical next action
  - Anticipate consequences of the proposed action
- For Text Input:
  - Note current cursor position
  - Consolidate repetitive actions (specify count for multiple keypresses)
  - Describe expected final text outcome

Your thought should be extremely brief, less than 50 words.

Use normalized coordinates (0.0 to 1.0) for all position-based commands.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "This is the state of my computer right now. Create a new project.",
                },
            ],
        },
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=chat_text,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    result = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    print("\nRaw model output:")
    print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
