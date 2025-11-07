"""
Generate grounding prompts for Qwen vision model desktop clicking tasks using Gemini API.

Reads JSON objects from wave-ui/data.jsonl, constructs natural user prompts using Gemini-2.5-Flash,
and writes the generated prompts to base/prompts/prompts.jsonl.

The prompts are designed for a vision model that will see the screen image, so they focus on
text-based identifiers and natural language actions (e.g., "click on the link to 'Product'").
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found. Install it with: pip install google-generativeai")
    sys.exit(1)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSON objects from a JSONL file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Warning:[/yellow] Skipping invalid JSON at line {line_num}: {e}")
                continue
            if isinstance(obj, dict):
                yield obj


def construct_prompt_input(obj: Dict[str, Any]) -> str:
    """
    Extract text-based fields from the JSON object to construct the input for Gemini.
    
    Only includes text-based identifiers that can be referenced in a natural prompt.
    Excludes visual/geometric information (bbox, resolution, description) since the
    vision model will see the image.
    
    Fields used:
    - instruction: The instruction text (if available)
    - name: Element name
    - type: Element type (button, link, input, etc.)
    - purpose: Purpose/action to perform
    - expectation: Expected outcome
    """
    parts = []

    # Element type - helps with natural language
    if obj.get("type"):
        parts.append(f"Element type: {obj['type']}")
    
    # Element name - additional context
    if obj.get("name"):
        parts.append(f"Element name: {obj['name']}")
    
    # Instruction - if it contains actionable text
    if obj.get("instruction"):
        parts.append(f"Instruction context: {obj['instruction']}")
    
    # Purpose - what action should be performed
    if obj.get("purpose"):
        parts.append(f"Action to perform: {obj['purpose']}")
    
    if obj.get("expectation"):
        parts.append(f"Expected outcome: {obj['expectation']}")

    return "\n".join(parts)


def generate_prompt_with_gemini(
    client: genai.GenerativeModel,
    obj: Dict[str, Any],
    max_retries: int = 3
) -> Optional[str]:
    """
    Generate a natural user prompt for a Qwen vision model using Gemini API.
    
    Creates a prompt that will be used with a screenshot image to train/ground a vision model
    for desktop clicking tasks. The prompt focuses on text-based identifiers and natural
    language actions.
    
    Args:
        client: The Gemini model instance
        obj: The JSON object containing element information (text-based fields only)
        max_retries: Maximum number of retry attempts
    
    Returns:
        The generated prompt text (e.g., "click on the link to 'Product'"), or None if generation failed
    """
    prompt_input = construct_prompt_input(obj)
    
    system_prompt = """You are creating a user prompt for training a Qwen vision model to perform desktop clicking tasks. 
The model will see a screenshot image, and your prompt should instruct it to click on a specific UI element.

Given information about a UI element, create a natural, concise user prompt that:
1. References the visible text on the element (from OCR) as the primary identifier
2. Uses natural, conversational language (e.g., "click on the link to 'Product'" or "click the 'Login' button")
3. Is action-oriented and clear about what to click
4. Does NOT reference bounding boxes, coordinates, or visual descriptions (the model sees the image)
5. Focuses on text-based identifiers that appear in the screenshot

Examples of good prompts:
- "click on the link to 'Product'"
- "click the 'Login' button"
- "click on 'Pricing'"
- "click the search box"
- "click on the 'Index' tab"

Generate only the prompt text itself, without any additional explanation, quotes, or formatting."""

    full_prompt = f"{system_prompt}\n\nElement Information:\n{prompt_input}\n\nGenerate the grounding prompt:"
    
    for attempt in range(max_retries):
        try:
            response = client.generate_content(full_prompt)
            if response and response.text:
                return response.text.strip()
            else:
                console.print(f"[yellow]Warning:[/yellow] Empty response from Gemini (attempt {attempt + 1})")
        except Exception as e:
            console.print(f"[red]Error:[/red] Gemini API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Brief delay before retry
    
    return None


def main():
    """Main function to process data and generate prompts."""
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] GEMINI_API_KEY environment variable not set.")
        console.print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        console.print("Or create a .env file with: GEMINI_API_KEY=your-api-key")
        sys.exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    # Try gemini-2.5-flash, fallback to gemini-2.0-flash-exp if not available
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        console.print("[yellow]Warning:[/yellow] gemini-2.5-flash not available, trying gemini-2.0-flash-exp")
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "wave-ui" / "data.jsonl"
    output_dir = project_root / "base" / "prompts"
    output_path = output_dir / "prompts.jsonl"
    
    # Check if data file exists
    if not data_path.exists():
        console.print(f"[red]Error:[/red] Data file not found: {data_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count total lines for progress
    total_lines = sum(1 for _ in stream_jsonl(str(data_path)))
    console.print(f"[green]Found[/green] {total_lines} records to process")
    
    # Process records
    success_count = 0
    error_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating prompts...", total=total_lines)
        
        with open(output_path, "w", encoding="utf-8") as out_file:
            for idx, obj in enumerate(stream_jsonl(str(data_path)), start=1):
                prompt_text = generate_prompt_with_gemini(model, obj)
                
                if prompt_text:
                    # Create output object with original data and generated prompt
                    output_obj = {
                        "id": idx,
                        "original": obj,
                        "prompt": prompt_text
                    }
                    out_file.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
                    success_count += 1
                else:
                    error_count += 1
                    console.print(f"[yellow]Warning:[/yellow] Failed to generate prompt for record {idx}")
                
                progress.update(task, advance=1)
                
                # Periodic status update
                if idx % 100 == 0:
                    console.print(f"[cyan]Processed[/cyan] {idx}/{total_lines} records...")
    
    # Summary
    console.print(f"\n[green]✓[/green] Successfully generated: {success_count} prompts")
    if error_count > 0:
        console.print(f"[yellow]⚠[/yellow] Failed: {error_count} prompts")
    console.print(f"[green]Output written to:[/green] {output_path}")


if __name__ == "__main__":
    main()

