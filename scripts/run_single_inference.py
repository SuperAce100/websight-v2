#!/usr/bin/env python3
"""
Run a single inference turn with the WebSight v2 grounded model on Apple Silicon.

The script loads `Asanshay/websight-v2-grounded`, runs the prompt against an image,
and prints the raw click command plus (optionally) scaled coordinates for the
current display resolution.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

DEFAULT_MODEL_ID = "Asanshay/websight-v2-grounded"
NORM_WIDTH = 1400
NORM_HEIGHT = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one inference turn with Asanshay/websight-v2-grounded using "
            "the Apple Silicon (MPS) backend."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the screenshot (PNG/JPG).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Instruction describing the target click, e.g. "click the login button".',
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local path (default: Asanshay/websight-v2-grounded).",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="Inference device. Defaults to mps and errors if MPS is unavailable.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype override (default: auto chooses float16 for MPS, bfloat16 otherwise).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate (default: 64).",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=None,
        help="Optional real screen width in pixels to rescale the prediction.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=None,
        help="Optional real screen height in pixels to rescale the prediction.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available. "
                "Ensure you're on Apple Silicon with torch>=2.1 and run `--device cpu` to override."
            )
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(choice: str, device: torch.device) -> torch.dtype:
    if choice == "auto":
        return torch.float16 if device.type == "mps" else torch.bfloat16
    if choice == "float16":
        return torch.float16
    return torch.bfloat16


def load_image(image_path: str) -> Image.Image:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def load_model_and_processor(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device.type == "mps":
        # Place weights directly on the Metal backend.
        model_kwargs["device_map"] = {"": "mps"}

    model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
    if device.type != "mps":
        model.to(device)
    model.eval()
    return model, processor


def run_generation(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    text = f"<image>\n{prompt}"
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return processor.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def parse_click_coordinates(output_text: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"pyautogui\.click\((\d+),\s*(\d+)\)", output_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def scale_coordinates(
    coords: Tuple[int, int],
    screen_width: Optional[int],
    screen_height: Optional[int],
) -> Optional[Tuple[int, int]]:
    if not coords or not screen_width or not screen_height:
        return None
    x_norm, y_norm = coords
    scaled_x = int(round(x_norm * (screen_width / NORM_WIDTH)))
    scaled_y = int(round(y_norm * (screen_height / NORM_HEIGHT)))
    return scaled_x, scaled_y


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"→ Loading {args.model_id} on {device} with dtype={dtype}...")
    model, processor = load_model_and_processor(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
    )

    image = load_image(args.image)
    output_text = run_generation(
        model=model,
        processor=processor,
        image=image,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nRaw model output:")
    print(output_text)

    coords = parse_click_coordinates(output_text)
    if coords:
        print(f"\nNormalized click (1400x800): x={coords[0]}, y={coords[1]}")
        scaled = scale_coordinates(coords, args.screen_width, args.screen_height)
        if scaled:
            print(
                f"Scaled to {args.screen_width}x{args.screen_height}: "
                f"x={scaled[0]}, y={scaled[1]}"
            )
        else:
            print("Provide --screen-width/--screen-height to rescale for your monitor.")
    else:
        print("\n⚠️ Could not parse `pyautogui.click(x, y)` from the output.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


