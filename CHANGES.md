# Changes: PyAutoGUI Output Format

## Summary

Updated the training pipeline to output PyAutoGUI commands instead of raw coordinates, and added a system prompt to guide the model's behavior.

## What Changed

### 1. Output Format
**Before:**
```
Input: click on the login button
Output: 945, 523
```

**After:**
```
System: You are a GUI automation assistant. Given an image and a user instruction, 
        output the exact pyautogui.click(x, y) command to execute the action. 
        Coordinates are normalized to 1400x800 resolution.

Input: click on the login button
Output: pyautogui.click(945, 523)
```

### 2. System Prompt Added

Every training sample now includes a system prompt that instructs the model:
> "You are a GUI automation assistant. Given an image and a user instruction, output the exact pyautogui.click(x, y) command to execute the action. Coordinates are normalized to 1400x800 resolution."

This ensures the model consistently outputs executable PyAutoGUI commands.

### 3. Files Modified

#### `scripts/transform_for_training.py`
- Updated `transform_record()` function to include system prompt
- Changed assistant response format from `"892, 336"` to `"pyautogui.click(892, 336)"`
- Updated docstring

#### `TRAINING_README.md`
- Updated "Output Format" section
- Modified inference example to show PyAutoGUI output
- Added note about executing commands with `exec()`

#### `SETUP_SUMMARY.md`
- Updated dataset format examples
- Modified inference code examples
- Updated output format description

#### `QUICKSTART.md`
- Added "Output Format" section explaining the new format
- Updated inference examples
- Modified "What This Does" section

## Benefits

1. **Directly Executable**: Output can be executed immediately with `exec(command)`
2. **Self-Documenting**: The command format is clear and unambiguous
3. **Consistent**: System prompt ensures uniform output format
4. **Automation-Ready**: Perfect for GUI automation workflows

## Usage After Training

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import pyautogui

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", device_map="auto"
)
model = PeftModel.from_pretrained(model, "saves/qwen3-vl-8b/lora/sft")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Get command
messages = [{"role": "user", "content": [
    {"type": "image", "image": "screenshot.png"},
    {"type": "text", "text": "click on the login button"}
]}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
command = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Example output: "pyautogui.click(945, 523)"
print(f"Generated: {command}")

# Execute the click
exec(command)  # Safely execute since we control the model output
```

## Data Format

Training data now includes three message roles:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a GUI automation assistant. Given an image and a user instruction, output the exact pyautogui.click(x, y) command to execute the action. Coordinates are normalized to 1400x800 resolution."
    },
    {
      "role": "user",
      "content": "<image>\nclick on the product link"
    },
    {
      "role": "assistant",
      "content": "pyautogui.click(892, 336)"
    }
  ],
  "images": ["/hai/scratch/asanshay/websight-v2/data/images/000000.png"]
}
```

## Migration Notes

If you already prepared data with the old format:
1. Simply re-run the data transformation script
2. The script will regenerate the training files with the new format

```bash
python scripts/transform_for_training.py
```

No changes needed to training configuration or SLURM scripts.

## Compatibility

- ✅ Compatible with LLaMA-Factory
- ✅ Compatible with Qwen3-VL-8B-Instruct
- ✅ Works with existing training configuration
- ✅ No additional dependencies required (PyAutoGUI is optional, for execution only)

