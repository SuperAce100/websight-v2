#!/usr/bin/env python3
"""
Transform agentnet dataset for UI-TARS 1.5 fine-tuning with Llama-Factory.

This script transforms agentnet trajectories into next-frame action prediction format:
- For each step i in a trajectory, creates a training example
- Input: Task instruction + full history (images, observations, thoughts, actions) up to step i
- Output: Next thought (CoT) + action + pyautogui code for step i+1

The format follows Llama-Factory's ShareGPT format with multi-turn conversations.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# ============================================================================
# Helper Functions: Message Formatting
# ============================================================================

def build_system_message(record: Dict) -> str:
    """
    Build system message with all task metadata.
    """
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

**Code:** Finally, output the action as PyAutoGUI code or the following functions:
- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}

Use normalized coordinates (0.0 to 1.0) for all position-based commands."""
    
    return system_prompt


def format_early_history_steps(history_steps: List[Dict]) -> str:
    """
    Format early history steps (before i-2) into a single assistant block.
    
    Each step includes step number and action. All combined into one assistant message.
    
    Args:
        history_steps: List of step data dictionaries with keys: step_idx, action
    
    Returns:
        Formatted assistant content string
    """
    assistant_parts = []
    
    for step_data in history_steps:
        step_idx = step_data["step_idx"]
        action = step_data.get("action", "")
        
        if action:
            assistant_parts.append(f"**Step {step_idx + 1}:** {action}")
    
    return "\n".join(assistant_parts) if assistant_parts else ""


def format_history_step_with_image(
    step_number: int,
    image_path: Optional[str],
    action: Optional[str],
) -> Tuple[str, str]:
    """
    Format a history step (i-1 or i-2) with image in Llama Factory format.
    
    User content: Only the image (with image_path tokenized in the content).
    Assistant content: Step number and action.
    
    Args:
        step_number: Step number (0-indexed)
        image_path: Path to the step's image
        action: Action description for this step
    
    Returns:
        Tuple of (user_content, assistant_content)
    """
    user_parts = []
    
    # User content: only the image (with image_path tokenized in the content)
    if image_path:
        # Include the image path directly in the user_content as a tokenized reference
        user_parts.append(f"<image>{image_path}</image>")
    
    user_content = "\n".join(user_parts) if user_parts else ""
    
    # Assistant content: step number and action
    assistant_parts = []
    
    if action:
        assistant_parts.append(f"**Step {step_number + 1}:** {action}")
    
    assistant_content = "\n".join(assistant_parts) if assistant_parts else ""
    
    return user_content, assistant_content


def format_current_step(
    natural_language_task: str,
    image_path: Optional[str],
    step_number: int,
    thought: Optional[str],
    action: Optional[str],
    code: Optional[str],
) -> Tuple[str, str, List[str]]:
    """
    Format the current step (the final step we're predicting from) in Llama Factory format.
    
    User content: Only the original task instruction (natural_language_task) + image.
    Assistant content: Step number, thought, action, and code.
    
    Args:
        natural_language_task: The original task instruction
        image_path: Path to the current step's image
        step_number: Step number (0-indexed)
        thought: Thought/reasoning text for this step
        action: Action description for this step
        code: PyAutoGUI code for this step
    
    Returns:
        Tuple of (user_content, assistant_content, [image_paths])
    """
    images = []
    user_parts = []
    
    # User content: only the task instruction + image
    if image_path:
        user_parts.append("<image>")
        images.append(image_path)
    
    if natural_language_task:
        user_parts.append(natural_language_task)
    
    user_content = "\n".join(user_parts) if user_parts else ""
    
    # Assistant content: step number, thought, action, code
    assistant_parts = []
    
    assistant_parts.append(f"**Step Number:** {step_number + 1}")
    
    if thought:
        assistant_parts.append(f"**Thought:** {thought}")
    
    if action:
        assistant_parts.append(f"**Action:** {action}")
    
    if code:
        assistant_parts.append(f"**Code:** {code}")
    
    assistant_content = "\n".join(assistant_parts) if assistant_parts else ""
    
    return user_content, assistant_content, images

# ============================================================================
# Helper Functions: Data Processing
# ============================================================================

def get_image_path(image: Optional[str], base_image_dir: Optional[str]) -> Optional[str]:
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


# ============================================================================
# Main Transformation Functions
# ============================================================================

def create_training_example(
    record: Dict,
    step_idx: int,
    base_image_dir: Optional[str] = None,
    include_reflection: bool = False,
) -> Optional[Dict]:
    """
    Create a training example for next-frame action prediction at step step_idx.
    
    Args:
        record: Original agentnet record
        step_idx: Index of the step to predict (0-indexed)
        base_image_dir: Base directory for images (if images are local)
        include_reflection: Whether to include reflection in the output
    
    Returns:
        Transformed record in Llama-Factory format, or None if invalid
    """
    traj = record.get("traj", [])
    if not isinstance(traj, list) or len(traj) <= step_idx + 1:
        return None
    
    # Get the step we're predicting
    next_step = traj[step_idx + 1]
    next_data = extract_step_data(next_step)
    if not next_data:
        return None
    
    # Build conversation history
    messages = []
    all_images = []
    
    # System message with all metadata
    messages.append({
        "role": "system",
        "content": build_system_message(record)
    })
    
    # Process history steps:
    # - Steps 0 to step_idx-3: Combined into single assistant block (no images)
    # - Step step_idx-2: User block (image only) + Assistant block (step + action)
    # - Step step_idx-1: User block (image only) + Assistant block (step + action)
    
    early_steps = []  # Steps 0 to step_idx-3
    
    # Collect early steps (0 to step_idx-3) for combined assistant block
    for i in range(step_idx - 2):
        step = traj[i]
        step_data = extract_step_data(step)
        if step_data and step_data.get("action"):
            early_steps.append({
                "step_idx": i,
                "action": step_data["action"]
            })
    
    # Add combined early steps as single assistant block
    if early_steps:
        early_history_content = format_early_history_steps(early_steps)
        if early_history_content:
            messages.append({
                "role": "assistant",
                "content": early_history_content
            })
    
    # Add step step_idx-2 with image (if it exists)
    if step_idx >= 2:
        step = traj[step_idx - 2]
        step_data = extract_step_data(step)
        if step_data:
            image_path = get_image_path(step_data["image"], base_image_dir)
            user_content, assistant_content = format_history_step_with_image(
                step_idx - 2,
                image_path,
                step_data.get("action")
            )
            
            if user_content:
                messages.append({
                    "role": "user",
                    "content": user_content
                })
                if image_path:
                    all_images.append(image_path)
            
            if assistant_content:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
    
    # Add step step_idx-1 with image (if it exists)
    if step_idx >= 1:
        step = traj[step_idx - 1]
        step_data = extract_step_data(step)
        if step_data:
            image_path = get_image_path(step_data["image"], base_image_dir)
            user_content, assistant_content = format_history_step_with_image(
                step_idx - 1,
                image_path,
                step_data.get("action")
            )
            
            if user_content:
                messages.append({
                    "role": "user",
                    "content": user_content
                })
                if image_path:
                    all_images.append(image_path)
            
            if assistant_content:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
    
    # Add current step (the one we're predicting from)
    current_step = traj[step_idx]
    current_data = extract_step_data(current_step)
    if current_data:
        # Get natural language task from record
        natural_language_task = record.get("natural_language_task", "")
        if not natural_language_task:
            natural_language_task = record.get("instruction", "")
        
        image_path = get_image_path(current_data["image"], base_image_dir)
        user_content, assistant_content, step_images = format_current_step(
            natural_language_task,
            image_path,
            step_idx,
            current_data["thought"],
            current_data["action"],
            current_data["code"]
        )
        
        if user_content or step_images:
            messages.append({
                "role": "user",
                "content": user_content
            })
            all_images.extend(step_images)
        
        if assistant_content:
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
    
    # Target output: next step's thought + action + code (and optionally reflection)
    next_thought = next_data.get("thought", "")
    next_action = next_data.get("action", "")
    next_code = next_data.get("code", "")
    next_reflection = next_data.get("reflection", "") if include_reflection else None
    
    if not (next_thought or next_action or next_code):
        return None
    
    return {
        "messages": messages,
        "images": all_images if all_images else None
    }


def transform_trajectory(
    record: Dict,
    base_image_dir: Optional[str] = None,
    include_reflection: bool = False,
    min_traj_length: int = 2,
) -> List[Dict]:
    """
    Transform a single trajectory record into multiple training examples.
    
    For a trajectory of length N, creates N-1 training examples (one for each
    step where we can predict the next action).
    
    Args:
        record: Original agentnet record
        base_image_dir: Base directory for images
        include_reflection: Whether to include reflection in output
        min_traj_length: Minimum trajectory length to process
    
    Returns:
        List of transformed training examples
    """
    traj = record.get("traj", [])
    if not isinstance(traj, list) or len(traj) < min_traj_length:
        return []
    
    examples = []
    # Create one example for each step where we can predict the next
    for i in range(len(traj) - 1):
        example = create_training_example(
            record, i, base_image_dir, include_reflection
        )
        if example:
            examples.append(example)
    
    return examples


def split_dataset(
    records: List[Dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        records: List of all records
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed
    
    Returns:
        Tuple of (train_records, val_records, test_records)
    """
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    
    test_records = shuffled[:test_size]
    val_records = shuffled[test_size:test_size + val_size]
    train_records = shuffled[test_size + val_size:]
    
    return train_records, val_records, test_records


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transform agentnet dataset for UI-TARS 1.5 training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input agentnet JSONL file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for transformed files",
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
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--min-traj-length",
        type=int,
        default=2,
        help="Minimum trajectory length to process",
    )
    parser.add_argument(
        "--max-examples-per-trajectory",
        type=int,
        default=None,
        help="Maximum examples to generate per trajectory (for testing)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Split into train/val/test
    train_records, val_records, test_records = split_dataset(
        records, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"Split: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test")
    
    # Transform each split
    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        output_path = output_dir / f"agentnet_{split_name}.jsonl"
        print(f"\nTransforming {split_name} set...")
        
        total_examples = 0
        skipped_trajectories = 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            for record in split_records:
                examples = transform_trajectory(
                    record,
                    args.base_image_dir,
                    args.include_reflection,
                    args.min_traj_length,
                )
                
                if not examples:
                    skipped_trajectories += 1
                    continue
                
                # Limit examples per trajectory if specified
                if args.max_examples_per_trajectory:
                    examples = examples[:args.max_examples_per_trajectory]
                
                for example in examples:
                    # Remove images field if empty
                    if example.get("images") is None or len(example["images"]) == 0:
                        example.pop("images", None)
                    
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    total_examples += 1
        
        print(f"  Saved {total_examples} examples to {output_path}")
        if skipped_trajectories > 0:
            print(f"  Skipped {skipped_trajectories} trajectories (too short or invalid)")
    
    # Print sample
    if records:
        print("\nSample transformed example:")
        sample_examples = transform_trajectory(
            records[0],
            args.base_image_dir,
            args.include_reflection,
            args.min_traj_length,
        )
        if sample_examples:
            print(json.dumps(sample_examples[0], indent=2, ensure_ascii=False))
    
    print("\n✓ Transformation complete!")


if __name__ == "__main__":
    main()
