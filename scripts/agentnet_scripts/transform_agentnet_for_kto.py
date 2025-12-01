#!/usr/bin/env python3
"""
Transform agentnet dataset for KTO (Kahneman-Tversky Optimization) reinforcement learning with LLaMA-Factory.

This script transforms agentnet trajectories into KTO format:
- Extracts each trajectory as a complete conversation
- Splits into positive examples (task_completed=True) and negative examples (task_completed=False)
- Formats as messages following LLaMA-Factory's ShareGPT format with labels

For KTO, we need preference pairs or labeled examples:
- Positive: task_completed=True (desired completions)
- Negative: task_completed=False (undesired completions)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_image_path(
    image: Optional[str], base_image_dir: Optional[str]
) -> Optional[str]:
    """
    Get full image path from image filename.

    Args:
        image: Image filename or path
        base_image_dir: Base directory for images (if images are local)

    Returns:
        Full image path string, or None if image is not provided
    """
    if not image:
        return None

    if base_image_dir:
        return str(Path(base_image_dir) / image)
    else:
        # Assume image is a relative path or URL
        return str(image)


def extract_step_data(step: Dict) -> Optional[Dict]:
    """
    Extract and validate data from a trajectory step.

    Args:
        step: Step dictionary from trajectory

    Returns:
        Dictionary with extracted step data, or None if invalid
    """
    if not isinstance(step, dict):
        return None

    value = step.get("value", {})
    if not isinstance(value, dict):
        return None

    return {
        "image": step.get("image"),
        "observation": value.get("observation"),
        "thought": value.get("thought"),
        "action": value.get("action"),
        "code": value.get("code"),
        "reflection": value.get("reflection", ""),
    }


def build_system_message(record: Dict) -> str:
    """
    Build system message with task metadata.

    Args:
        record: Original agentnet record

    Returns:
        System message string
    """
    instruction = record.get("instruction", "")
    natural_language_task = record.get("natural_language_task", "")
    actual_task = record.get("actual_task", "")

    # Use natural_language_task if available, otherwise instruction
    task_description = natural_language_task or instruction

    system_prompt = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

For each step, provide your response in this format:

**Thought:** 
- Step by Step Progress Assessment:
  - Analyze completed task parts and their contribution to the overall goal
  - Reflect on potential errors, unexpected results, or obstacles
  - If previous action was incorrect, predict a logical recovery step
- Next Action Analysis:
  - List possible next actions based on current state
  - Evaluate options considering current state and previous actions
  - Propose most logical next action
  - Anticipate consequences of the proposed action
- For Text Input:
  - Note current cursor position
  - Consolidate repetitive actions (specify count for multiple keypresses)
  - Describe expected final text outcome
- Use first-person perspective in reasoning

**Action:** Provide clear, concise, and actionable instructions:
- If the action involves interacting with a specific target:
  - Describe target explicitly without using coordinates
  - Specify element names when possible
  - Describe features (shape, color, position) if name unavailable
  - For window control buttons, identify correctly (minimize, maximize, close)
- If the action involves keyboard actions like 'press', 'write', 'hotkey':
  - Consolidate repetitive keypresses with count
  - Specify expected text outcome for typing actions

**Code:** Finally, output the action as a PyAutoGUI command (e.g. pyautogui.click(x, y), etc.) or the following functions:
- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}

Use normalized coordinates (0.0 to 1.0) for all position-based commands."""

    return system_prompt


def format_assistant_response(
    step_data: Dict, step_number: int, include_reflection: bool = False
) -> str:
    """
    Format assistant response for a trajectory step.

    Args:
        step_data: Extracted step data dictionary
        step_number: Step number (0-indexed)
        include_reflection: Whether to include reflection

    Returns:
        Formatted assistant content string
    """
    parts = []
    parts.append(f"**Step Number:** {step_number + 1}")

    thought = step_data.get("thought", "")
    if thought:
        parts.append(f"**Thought:** {thought}")

    action = step_data.get("action", "")
    if action:
        parts.append(f"**Action:** {action}")

    code = step_data.get("code", "")
    if code:
        parts.append(f"**Code:** {code}")

    if include_reflection:
        reflection = step_data.get("reflection", "")
        if reflection:
            parts.append(f"**Reflection:** {reflection}")

    return "\n".join(parts)


def format_user_message(
    task_description: str,
    image_path: Optional[str],
    observation: Optional[str],
    is_first_step: bool = False,
) -> str:
    """
    Format user message for a trajectory step.

    Args:
        task_description: The task instruction
        image_path: Path to the step's image
        observation: Observation text for this step
        is_first_step: Whether this is the first step in the trajectory

    Returns:
        Formatted user content string
    """
    parts = []

    # Include image tag if available
    if image_path:
        parts.append("<image>")

    # Include task description on first step
    if is_first_step and task_description:
        parts.append(task_description)

    # Include observation if available
    if observation:
        parts.append(f"**Observation:** {observation}")

    return "\n".join(parts)


def transform_trajectory_to_kto_example(
    record: Dict,
    base_image_dir: Optional[str] = None,
    include_reflection: bool = False,
    min_traj_length: int = 1,
) -> Optional[Dict]:
    """
    Transform a single trajectory record into a KTO training example.

    Args:
        record: Original agentnet record
        base_image_dir: Base directory for images (if images are local)
        include_reflection: Whether to include reflection in output
        min_traj_length: Minimum trajectory length to process

    Returns:
        Transformed record in KTO format with label, or None if invalid
    """
    traj = record.get("traj", [])
    if not isinstance(traj, list) or len(traj) < min_traj_length:
        return None

    # Check task_completed to determine label
    task_completed = record.get("task_completed", False)
    
    # Get task description
    instruction = record.get("instruction", "")
    natural_language_task = record.get("natural_language_task", "")
    task_description = natural_language_task or instruction

    # Build conversation
    messages = []
    all_images = []

    # System message
    messages.append({"role": "system", "content": build_system_message(record)})

    # Add trajectory steps as conversation turns
    for i, step in enumerate(traj):
        step_data = extract_step_data(step)
        if not step_data:
            continue

        image_path = get_image_path(step_data["image"], base_image_dir)
        observation = step_data.get("observation", "")
        
        # User message: task (on first step) + image + observation
        is_first_step = (i == 0)
        user_content = format_user_message(
            task_description, image_path, observation, is_first_step
        )
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
            if image_path:
                all_images.append(image_path)

        # Assistant message: thought + action + code
        assistant_content = format_assistant_response(
            step_data, i, include_reflection
        )
        
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

    # Must have at least one user-assistant pair
    if len(messages) < 3:  # system + user + assistant
        return None

    # Build KTO example
    kto_example = {
        "messages": messages,
        "label": "true" if task_completed else "false",  # 1 for positive, 0 for negative
    }

    # Add images if available
    if all_images:
        kto_example["images"] = all_images

    return kto_example


def main():
    parser = argparse.ArgumentParser(
        description="Transform agentnet dataset for KTO reinforcement learning"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input agentnet JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path for KTO examples",
    )
    parser.add_argument(
        "--base-image-dir",
        type=str,
        default=None,
        help="Base directory for images (if images are local files)",
    )
    parser.add_argument(
        "--include-reflection",
        action="store_true",
        help="Include reflection in assistant output",
    )
    parser.add_argument(
        "--min-traj-length",
        type=int,
        default=1,
        help="Minimum trajectory length to process",
    )
    parser.add_argument(
        "--split-by-label",
        action="store_true",
        help="Split output into separate positive and negative files",
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Reading dataset from {input_path}...")

    # Read all records
    records = []
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue

    print(f"Loaded {len(records)} trajectory records")

    # Transform records
    positive_examples = []
    negative_examples = []
    skipped = 0

    for record in records:
        example = transform_trajectory_to_kto_example(
            record,
            args.base_image_dir,
            args.include_reflection,
            args.min_traj_length,
        )

        if not example:
            skipped += 1
            continue

        # Separate by label
        if example["label"] == "true":
            positive_examples.append(example)
        else:
            negative_examples.append(example)

    print(f"\nTransformed examples:")
    print(f"  Positive (task_completed=True): {len(positive_examples)}")
    print(f"  Negative (task_completed=False): {len(negative_examples)}")
    print(f"  Skipped: {skipped}")

    # Write output
    if args.split_by_label:
        # Split into separate files
        output_dir = output_path.parent
        output_stem = output_path.stem
        output_suffix = output_path.suffix

        positive_output = output_dir / f"{output_stem}_positive{output_suffix}"
        negative_output = output_dir / f"{output_stem}_negative{output_suffix}"

        # Write positive examples
        if positive_examples:
            with open(positive_output, "w", encoding="utf-8") as f:
                for example in positive_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            print(f"\nSaved {len(positive_examples)} positive examples to {positive_output}")

        # Write negative examples
        if negative_examples:
            with open(negative_output, "w", encoding="utf-8") as f:
                for example in negative_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            print(f"Saved {len(negative_examples)} negative examples to {negative_output}")
    else:
        # Write all examples to single file
        all_examples = positive_examples + negative_examples
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        print(f"\nSaved {len(all_examples)} examples to {output_path}")

    # Print sample
    if positive_examples or negative_examples:
        sample = positive_examples[0] if positive_examples else negative_examples[0]
        print("\nSample KTO example:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))

    print("\n✓ Transformation complete!")


if __name__ == "__main__":
    main()

