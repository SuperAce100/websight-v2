"""
Generate grounding prompts for Qwen vision model desktop clicking tasks using Gemini API.

Reads JSON objects from wave-ui/data.jsonl, constructs natural user prompts using Gemini-2.5-Flash-Lite,
and writes the generated prompts to base/prompts/prompts.jsonl.

The prompts are designed for a vision model that will see the screen image, so they focus on
text-based identifiers and natural language actions (e.g., "click on the link to 'Product'").

Uses Gemini 2.5 Flash-Lite with rate limits: 4,000 RPM, 4M TPM, no RPD limit.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, List, Tuple
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found. Install it with: pip install google-generativeai")
    sys.exit(1)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Rate limiting configuration
# Gemini 2.5 Flash-Lite: 4,000 RPM, 4M TPM, no RPD limit
# Default to 4000 RPM (66.67 RPS) for Flash-Lite
REQUESTS_PER_MINUTE = int(os.getenv("GEMINI_RPM", "4000"))  # Default for Flash-Lite
REQUESTS_PER_SECOND = REQUESTS_PER_MINUTE / 60.0
MIN_INTERVAL = 1.0 / REQUESTS_PER_SECOND  # Minimum time between requests
# Allow higher concurrency for Flash-Lite (up to 100 concurrent requests)
MAX_CONCURRENT = min(int(os.getenv("GEMINI_MAX_CONCURRENT", "50")), REQUESTS_PER_MINUTE // 40)


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


class RateLimiter:
    """Rate limiter to respect API quotas."""
    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until we can make a request."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()


async def generate_prompt_with_gemini_async(
    model: genai.GenerativeModel,
    obj: Dict[str, Any],
    rate_limiter: RateLimiter,
    idx: int,
    max_retries: int = 3
) -> Tuple[int, Optional[str]]:
    """
    Generate a natural user prompt for a Qwen vision model using Gemini API (async).
    
    Args:
        model: The Gemini model instance
        obj: The JSON object containing element information
        rate_limiter: Rate limiter instance
        idx: Record index for tracking
        max_retries: Maximum number of retry attempts
    
    Returns:
        Tuple of (index, generated prompt text or None)
    """
    prompt_input = construct_prompt_input(obj)
    
    system_prompt = """You are creating a user prompt for training a Qwen vision model to perform desktop clicking tasks. 
The model will see a screenshot image, and your prompt should instruct it to click on a specific UI element.

Given information about a UI element, create a natural, concise user prompt that:
1. Synthesizes the provided information into a natural user prompt
2. Uses natural, conversational language (e.g., "click on the link to 'Product'" or "click the 'Login' button")
3. Is action-oriented and clear about what to click
4. Focuses on text-based identifiers that would appear in the screenshot

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
            # Rate limit before making request
            await rate_limiter.acquire()
            
            # Make the API call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            # Use functools.partial or a proper function to avoid closure issues
            def make_request():
                return model.generate_content(full_prompt)
            response = await loop.run_in_executor(None, make_request)
            
            if response and response.text:
                return (idx, response.text.strip())
            else:
                if attempt == max_retries - 1:
                    console.print(f"[yellow]Warning:[/yellow] Empty response for record {idx}")
        except Exception as e:
            if attempt == max_retries - 1:
                console.print(f"[red]Error:[/red] Failed to generate prompt for record {idx}: {e}")
            else:
                await asyncio.sleep(1)  # Brief delay before retry
    
    return (idx, None)


async def process_batch(
    model: genai.GenerativeModel,
    batch: List[Tuple[int, Dict[str, Any]]],
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore
) -> List[Tuple[int, Optional[str]]]:
    """Process a batch of records concurrently."""
    async def process_one(idx_obj_pair: Tuple[int, Dict[str, Any]]):
        idx, obj = idx_obj_pair
        async with semaphore:
            return await generate_prompt_with_gemini_async(model, obj, rate_limiter, idx)
    
    tasks = [process_one(pair) for pair in batch]
    return await asyncio.gather(*tasks)


async def main_async():
    """Main async function to process data and generate prompts with batching."""
    # Get API key from environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] GEMINI_API_KEY environment variable not set.")
        console.print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        console.print("Or create a .env file with: GEMINI_API_KEY=your-api-key")
        sys.exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    # Use gemini-2.5-flash-lite (4k RPM, 4M TPM, no RPD limit)
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
    except Exception:
        console.print("[yellow]Warning:[/yellow] gemini-2.5-flash-lite not available, trying gemini-2.5-flash")
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
    
    # Load all records into memory (for batching)
    console.print("[cyan]Loading records...[/cyan]")
    records = list(enumerate(stream_jsonl(str(data_path)), start=1))
    total_lines = len(records)
    console.print(f"[green]Found[/green] {total_lines} records to process")
    console.print(f"[cyan]Rate limit:[/cyan] {REQUESTS_PER_MINUTE} RPM ({REQUESTS_PER_SECOND:.2f} RPS)")
    console.print(f"[cyan]Concurrency:[/cyan] {MAX_CONCURRENT} concurrent requests")
    
    # Initialize rate limiter and semaphore
    rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Batch size - process in chunks to manage memory
    batch_size = MAX_CONCURRENT * 10  # Process larger batches
    success_count = 0
    error_count = 0
    
    # Store results temporarily (index -> result)
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating prompts...", total=total_lines)
        
        # Process in batches
        for batch_start in range(0, total_lines, batch_size):
            batch_end = min(batch_start + batch_size, total_lines)
            batch = records[batch_start:batch_end]
            
            # Process batch concurrently
            batch_results = await process_batch(model, batch, rate_limiter, semaphore)
            
            # Store results
            for idx, prompt_text in batch_results:
                results[idx] = prompt_text
                if prompt_text:
                    success_count += 1
                else:
                    error_count += 1
                
                progress.update(task, advance=1)
            
            # Periodic status update
            if batch_end % 1000 == 0 or batch_end == total_lines:
                console.print(f"[cyan]Processed[/cyan] {batch_end}/{total_lines} records... "
                            f"({success_count} success, {error_count} errors)")
    
    # Write results to file in order
    console.print("[cyan]Writing results to file...[/cyan]")
    with open(output_path, "w", encoding="utf-8") as out_file:
        for idx, obj in records:
            prompt_text = results.get(idx)
            if prompt_text:
                output_obj = {
                    "id": idx,
                    "original": obj,
                    "prompt": prompt_text
                }
                out_file.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
    
    # Summary
    console.print(f"\n[green]✓[/green] Successfully generated: {success_count} prompts")
    if error_count > 0:
        console.print(f"[yellow]⚠[/yellow] Failed: {error_count} prompts")
    console.print(f"[green]Output written to:[/green] {output_path}")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

