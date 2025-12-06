"""
Local UI-TARS Agent for OSWorld evaluation.

This agent loads UI-TARS 1.5 directly from HuggingFace via transformers
(no API required).

Usage:
    python run_multienv_my_uitars.py \
        --model Asanshay/websight-2-7B-kto \
        --provider_name aws \
        --headless \
        --max_steps 15
"""

import ast
import base64
import logging
import math
import re
from io import BytesIO
from typing import Dict, List, Tuple, Any, Optional

import torch
from PIL import Image

logger = logging.getLogger("desktopenv.agent")

# UI-TARS action space and prompts
UITARS_ACTION_SPACE = """click(start_box='(x,y)')
left_double(start_box='(x,y)')
right_single(start_box='(x,y)')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
hotkey(key='key1 key2')
type(content='text')
scroll(start_box='(x,y)', direction='up|down|left|right')
wait()
finished()"""

UITARS_SYSTEM_PROMPT = """You are a GUI automation agent. You look at screenshots and output actions to complete tasks.

SCREEN: 2560x1440 pixels. Coordinates are (x,y) where x=0-2560, y=0-1440.

OUTPUT FORMAT (strict):
Thought: [one sentence describing what you see and what to do next]
Action: [exactly one function call from the action space below]

ACTION SPACE:
{action_space}

KEYBOARD SHORTCUTS (use hotkey):
- hotkey(key='ctrl t') - new tab
- hotkey(key='ctrl w') - close tab  
- hotkey(key='ctrl shift t') - restore closed tab
- hotkey(key='ctrl l') - focus address bar
- hotkey(key='ctrl a') - select all
- hotkey(key='ctrl c') - copy
- hotkey(key='ctrl v') - paste
- hotkey(key='alt left') - back
- hotkey(key='f5') - refresh

EXAMPLES:
Thought: I see a search box at coordinates (800, 100). I will click it.
Action: click(start_box='(800,100)')

Thought: I need to type the search query and submit.
Action: type(content='my search query\\n')

Thought: The tab was closed. I will restore it with Ctrl+Shift+T.
Action: hotkey(key='ctrl shift t')

Thought: I need to scroll down to see more content.
Action: scroll(start_box='(1280,720)', direction='down')

Thought: The task is complete.
Action: finished()

RULES:
- Action must be EXACTLY a function call, never natural language
- Always include coordinates for click/scroll actions
- Use hotkey() for keyboard shortcuts instead of clicking

TASK: {instruction}
"""

# Constants
FINISH_WORD = "finished"
WAIT_WORD = "wait"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Resize image dimensions to be divisible by factor and within pixel limits."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def parse_action(action_str: str) -> Optional[Dict]:
    """Parse a single action string into function name and args."""
    try:
        node = ast.parse(action_str, mode="eval")
        if not isinstance(node, ast.Expression):
            return None

        call = node.body
        if not isinstance(call, ast.Call):
            return None

        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            return None

        kwargs = {}
        for kw in call.keywords:
            if isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif isinstance(kw.value, ast.Str):
                kwargs[kw.arg] = kw.value.s

        return {"function": func_name, "args": kwargs}
    except Exception as e:
        logger.debug(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text: str) -> str:
    """Escape unescaped single quotes."""
    return re.sub(r"(?<!\\)'", r"\\'", text)


def parse_uitars_response(
    text: str, image_height: int, image_width: int
) -> Tuple[str, List[Dict]]:
    """
    Parse UI-TARS model response into structured actions.

    Searches the ENTIRE response for any valid action patterns, not just after "Action:".
    This handles cases where the model outputs actions in non-standard formats.

    Returns:
        Tuple of (thought, list of action dicts)
    """
    text = text.strip()

    # Extract thought
    thought = ""
    thought_match = re.search(r"Thought:\s*(.+?)(?=\s*Action:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # List of action patterns to search for anywhere in the response
    ACTION_PATTERNS = [
        r"(click\s*\(\s*start_box\s*=\s*['\"].*?['\"]\s*\))",
        r"(left_double\s*\(\s*start_box\s*=\s*['\"].*?['\"]\s*\))",
        r"(right_single\s*\(\s*start_box\s*=\s*['\"].*?['\"]\s*\))",
        r"(drag\s*\(\s*start_box\s*=\s*['\"].*?['\"]\s*,\s*end_box\s*=\s*['\"].*?['\"]\s*\))",
        r"(scroll\s*\(\s*start_box\s*=\s*['\"].*?['\"]\s*,\s*direction\s*=\s*['\"].*?['\"]\s*\))",
        r"(scroll\s*\(\s*direction\s*=\s*['\"].*?['\"]\s*\))",
        r"(hotkey\s*\(\s*key\s*=\s*['\"].*?['\"]\s*\))",
        r"(type\s*\(\s*content\s*=\s*['\"].*?['\"]\s*\))",
        r"(wait\s*\(\s*\))",
        r"(finished\s*\(\s*\))",
    ]

    action_str = None

    # First try the standard extraction methods
    if "Code:" in text:
        action_str = text.split("Code:")[-1].strip()
    elif "Action:" in text:
        action_str = text.split("Action:")[-1].strip()

    # Try parsing the extracted action string first
    if action_str:
        # Handle type() with special characters
        if "type(content" in action_str:
            pattern = r"type\(content='(.*?)'\)"
            match = re.search(pattern, action_str)
            if match:
                content = escape_single_quotes(match.group(1))
                action_str = f"type(content='{content}')"

        parsed = parse_action(action_str.replace("\n", "\\n").strip())
        if parsed is not None:
            action_type = parsed["function"]
            params = parsed["args"]
            action_inputs = _process_action_params(params)
            return thought, [
                {
                    "thought": thought,
                    "action_type": action_type,
                    "action_inputs": action_inputs,
                }
            ]

    # If standard parsing failed, search the ENTIRE response for any action pattern
    for pattern in ACTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            action_str = match.group(1).strip()
            logger.debug(f"Found action pattern in response: {action_str}")

            # Handle type() with special characters
            if "type(content" in action_str.lower():
                type_pattern = r"type\(content='(.*?)'\)"
                type_match = re.search(type_pattern, action_str, re.IGNORECASE)
                if type_match:
                    content = escape_single_quotes(type_match.group(1))
                    action_str = f"type(content='{content}')"

            parsed = parse_action(action_str.replace("\n", "\\n").strip())
            if parsed is not None:
                action_type = parsed["function"]
                params = parsed["args"]
                action_inputs = _process_action_params(params)
                return thought, [
                    {
                        "thought": thought,
                        "action_type": action_type,
                        "action_inputs": action_inputs,
                    }
                ]

    # Check for terminal actions anywhere in text
    if re.search(r"\bfinished\s*\(\s*\)", text, re.IGNORECASE):
        return thought, [
            {"thought": thought, "action_type": "finished", "action_inputs": {}}
        ]
    if re.search(r"\bwait\s*\(\s*\)", text, re.IGNORECASE):
        return thought, [
            {"thought": thought, "action_type": "wait", "action_inputs": {}}
        ]

    return thought, []


def _process_action_params(params: Dict) -> Dict:
    """Process action parameters, normalizing coordinates."""
    action_inputs = {}
    for param_name, param in params.items():
        if param == "":
            continue
        param = str(param).strip()
        action_inputs[param_name] = param

        if "start_box" in param_name or "end_box" in param_name:
            # Extract coordinates from format (x,y) or <|box_start|>(x,y)<|box_end|>
            coord_match = re.search(r"\((\d+),\s*(\d+)\)", param)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                # Normalize coordinates
                # IMPORTANT: Set these to your model's training resolution
                x_norm = x / 2560.0
                y_norm = y / 1440.0
                action_inputs[param_name] = str([x_norm, y_norm, x_norm, y_norm])
    return action_inputs


def actions_to_pyautogui(
    actions: List[Dict], image_height: int, image_width: int
) -> str:
    """Convert parsed actions to pyautogui code."""
    if not actions:
        return "WAIT"

    action = actions[0]
    action_type = action.get("action_type", "")
    action_inputs = action.get("action_inputs", {})

    # Terminal actions
    if action_type == FINISH_WORD:
        return "DONE"
    if action_type == WAIT_WORD:
        return "WAIT"
    if action_type == CALL_USER:
        return "FAIL"

    pyautogui_code = "import pyautogui\nimport time\n"

    if action.get("thought"):
        pyautogui_code += f"'''\nThought: {action['thought']}\n'''\n"

    if action_type == "hotkey":
        key = action_inputs.get("key", "")
        if key:
            keys = key.split()
            keys = ["ctrl" if k == "control" else k for k in keys]
            pyautogui_code += (
                f"\npyautogui.hotkey({', '.join([repr(k) for k in keys])})"
            )

    elif action_type == "type":
        content = action_inputs.get("content", "")
        content = escape_single_quotes(content)
        if content:
            stripped = content.rstrip("\\n").rstrip("\n")
            pyautogui_code += f"\nimport pyperclip"
            pyautogui_code += f"\npyperclip.copy('{stripped}')"
            pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
            pyautogui_code += f"\ntime.sleep(0.5)"
            if content.endswith("\n") or content.endswith("\\n"):
                pyautogui_code += f"\npyautogui.press('enter')"

    elif action_type in ["click", "left_single"]:
        start_box = action_inputs.get("start_box")
        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.click({x}, {y})"

    elif action_type == "left_double":
        start_box = action_inputs.get("start_box")
        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.doubleClick({x}, {y})"

    elif action_type == "right_single":
        start_box = action_inputs.get("start_box")
        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"

    elif action_type == "drag":
        start_box = action_inputs.get("start_box")
        end_box = action_inputs.get("end_box")
        if start_box and end_box:
            start_coords = eval(start_box)
            end_coords = eval(end_box)
            sx = round(float((start_coords[0] + start_coords[2]) / 2) * image_width)
            sy = round(float((start_coords[1] + start_coords[3]) / 2) * image_height)
            ex = round(float((end_coords[0] + end_coords[2]) / 2) * image_width)
            ey = round(float((end_coords[1] + end_coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.moveTo({sx}, {sy})"
            pyautogui_code += f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)"

    elif action_type == "scroll":
        direction = action_inputs.get("direction", "down")
        start_box = action_inputs.get("start_box")

        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            if "up" in direction.lower():
                pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
            else:
                pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"
        else:
            if "up" in direction.lower():
                pyautogui_code += f"\npyautogui.scroll(5)"
            else:
                pyautogui_code += f"\npyautogui.scroll(-5)"

    elif action_type == "press_enter":
        pyautogui_code += f"\npyautogui.press('return')"

    elif action_type == "press":
        # Generic press(key) action
        key = action_inputs.get("key", "")
        if key:
            # Normalize common key names
            key = key.lower().replace("enter", "return").replace("esc", "escape")
            pyautogui_code += f"\npyautogui.press('{key}')"

    elif action_type == "middleClick":
        start_box = action_inputs.get("start_box")
        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.click({x}, {y}, button='middle')"

    elif action_type == "tripleClick":
        start_box = action_inputs.get("start_box")
        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.tripleClick({x}, {y})"

    elif action_type == "moveTo":
        start_box = action_inputs.get("start_box")
        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

    elif action_type == "dragTo":
        # Alternative drag format: dragTo(x, y) instead of drag(start_box, end_box)
        end_box = action_inputs.get("end_box") or action_inputs.get("start_box")
        if end_box:
            coords = eval(end_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            pyautogui_code += f"\npyautogui.dragTo({x}, {y}, duration=1.0)"

    elif action_type == "hscroll":
        # Horizontal scroll
        direction = action_inputs.get("direction", "right")
        start_box = action_inputs.get("start_box")
        if start_box:
            coords = eval(start_box)
            x = round(float((coords[0] + coords[2]) / 2) * image_width)
            y = round(float((coords[1] + coords[3]) / 2) * image_height)
            if "left" in direction.lower():
                pyautogui_code += f"\npyautogui.hscroll(-5, x={x}, y={y})"
            else:
                pyautogui_code += f"\npyautogui.hscroll(5, x={x}, y={y})"
        else:
            if "left" in direction.lower():
                pyautogui_code += f"\npyautogui.hscroll(-5)"
            else:
                pyautogui_code += f"\npyautogui.hscroll(5)"

    elif action_type == "write":
        # Alternative to type()
        content = action_inputs.get("text", "") or action_inputs.get("content", "")
        content = escape_single_quotes(content)
        if content:
            stripped = content.rstrip("\\n").rstrip("\n")
            pyautogui_code += f"\nimport pyperclip"
            pyautogui_code += f"\npyperclip.copy('{stripped}')"
            pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
            pyautogui_code += f"\ntime.sleep(0.5)"
            if content.endswith("\n") or content.endswith("\\n"):
                pyautogui_code += f"\npyautogui.press('return')"

    elif action_type == "terminate":
        # terminate('success') or terminate('failure')
        status = action_inputs.get("status", "success")
        if "fail" in str(status).lower():
            return "FAIL"
        else:
            return "DONE"

    else:
        pyautogui_code += f"\n# Unknown action: {action_type}"

    return pyautogui_code


class MyUITarsAgent:
    """
    Local UI-TARS Agent that loads the model via transformers.
    No API required - runs inference locally on GPU/CPU.
    """

    # UI-TARS 1.5 is based on Qwen2.5-VL, so we use its processor
    BASE_MODEL_FOR_PROCESSOR = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        model: str = "Asanshay/websight-2-7B-kto",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 5,
        screen_width: int = 1920,
        screen_height: int = 1080,
        device: str = "auto",
        dtype: str = "auto",
        language: str = "English",
        **kwargs,
    ):
        """
        Initialize UI-TARS Agent.

        Args:
            model: HuggingFace model ID (default: Asanshay/websight-2-7B-kto)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            action_space: Must be "pyautogui"
            observation_type: Must be "screenshot"
            max_trajectory_length: Max history length
            screen_width: Screen width for coordinate scaling
            screen_height: Screen height for coordinate scaling
            device: Device ("auto", "cuda", "mps", "cpu")
            dtype: Data type ("auto", "float16", "bfloat16")
            language: Language for thought output
        """
        self.model_id = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.language = language

        # Trajectory storage
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []

        # Stuck detection
        self.consecutive_waits = 0
        self.last_action_type = None

        # Load model
        self._load_model(device, dtype)

        logger.info(f"MyUITarsAgent initialized with model: {model}")

    def _load_model(self, device: str, dtype: str):
        """Load model and processor."""
        from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Determine dtype
        if dtype == "auto":
            if self.device.type == "mps":
                self.dtype = torch.float16
            elif self.device.type == "cuda":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32
        elif dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        logger.info(f"Loading model on {self.device} with dtype {self.dtype}")

        # Load processor - UI-TARS 1.5 uses Qwen2.5-VL architecture
        # Try loading from model first, fall back to base Qwen2.5-VL
        try:
            self.processor = Qwen2_5_VLProcessor.from_pretrained(self.model_id)
            logger.info(f"Loaded processor from model: {self.model_id}")
        except Exception as e:
            logger.info(f"Could not load processor from {self.model_id}: {e}")
            logger.info(
                f"Loading processor from base model: {self.BASE_MODEL_FOR_PROCESSOR}"
            )
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                self.BASE_MODEL_FOR_PROCESSOR
            )

        # Load model
        model_kwargs = {
            "torch_dtype": self.dtype,
        }

        if self.device.type in ["cuda", "mps"]:
            model_kwargs["device_map"] = {"": self.device.type}

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **model_kwargs
        )

        if self.device.type == "cpu":
            self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to fit within model's pixel limits."""
        width, height = image.size

        if width * height > MAX_PIXELS:
            resize_factor = math.sqrt(MAX_PIXELS / (width * height))
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            image = image.resize((new_width, new_height))

        if width * height < MIN_PIXELS:
            resize_factor = math.sqrt(MIN_PIXELS / (width * height))
            new_width = math.ceil(width * resize_factor)
            new_height = math.ceil(height * resize_factor)
            image = image.resize((new_width, new_height))

        return image

    def _build_prompt(self, instruction: str, add_recovery_hint: bool = False) -> str:
        """Build the UI-TARS prompt."""
        prompt = UITARS_SYSTEM_PROMPT.format(
            action_space=UITARS_ACTION_SPACE, instruction=instruction
        )

        if add_recovery_hint:
            prompt += """

WARNING: Previous actions failed. You MUST output a valid action now.
- Your Action line MUST be exactly like: click(start_box='(x,y)') or hotkey(key='ctrl t')
- Do NOT write sentences in Action. Only function calls.
- Try a keyboard shortcut if clicking isn't working.
- Look carefully at the screenshot for clickable elements.
"""

        return prompt

    def _build_messages(self, instruction: str, image: Image.Image) -> List[Dict]:
        """Build conversation messages for the model."""
        prompt = self._build_prompt(instruction)

        # For UI-TARS, we use a simple format
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}],
            }
        ]

        # Add history if available
        for i, (prev_response, prev_image) in enumerate(
            zip(
                self.history_responses[-self.max_trajectory_length :],
                self.history_images[-self.max_trajectory_length :],
            )
        ):
            # Insert previous turns before current
            pass  # For simplicity, we don't use multi-turn history in this version

        return messages

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        """
        Predict the next action based on current observation.

        Args:
            instruction: Task instruction
            obs: Dict with "screenshot" (bytes) and optionally "accessibility_tree"

        Returns:
            Tuple of (response_text, list_of_pyautogui_actions)
        """
        assert len(self.observations) == len(self.actions) == len(self.thoughts)

        # Process screenshot
        screenshot_bytes = obs.get("screenshot")
        if screenshot_bytes is None:
            logger.error("No screenshot in observation")
            return "Error: No screenshot", ["FAIL"]

        image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
        image = self._resize_image(image)

        # Store observation
        base64_screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")
        self.observations.append(
            {
                "screenshot": base64_screenshot,
                "accessibility_tree": obs.get("accessibility_tree"),
            }
        )
        self.history_images.append(screenshot_bytes)

        # Check if stuck (3+ consecutive WAITs)
        add_recovery_hint = self.consecutive_waits >= 2
        if add_recovery_hint:
            logger.info(
                f"Agent appears stuck ({self.consecutive_waits} consecutive WAITs), adding recovery hint"
            )

        # Build prompt
        prompt = self._build_prompt(instruction, add_recovery_hint=add_recovery_hint)

        # Build messages in Qwen2-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs - pass image directly to processor
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                )

            # Decode only the generated part (exclude input tokens)
            generated_ids = [
                output_ids[i][len(inputs.input_ids[i]) :]
                for i in range(len(output_ids))
            ]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            self.thoughts.append("")
            self.actions.append(["FAIL"])
            return f"Error: {e}", ["FAIL"]

        logger.info(f"Model response: {response}")

        # Parse response
        try:
            thought, parsed_actions = parse_uitars_response(
                response, self.screen_height, self.screen_width
            )

            if not parsed_actions:
                logger.warning("No actions parsed from response")
                self.thoughts.append(response)
                self.actions.append(["WAIT"])
                self.history_responses.append(response)
                self.consecutive_waits += 1
                return response, ["WAIT"]

            # Check for terminal actions
            action_type = parsed_actions[0].get("action_type", "")
            if action_type == FINISH_WORD:
                self.thoughts.append(thought)
                self.actions.append(["DONE"])
                self.history_responses.append(response)
                return response, ["DONE"]

            if action_type == WAIT_WORD:
                self.thoughts.append(thought)
                self.actions.append(["WAIT"])
                self.history_responses.append(response)
                self.consecutive_waits += 1
                return response, ["WAIT"]

            # Convert to pyautogui
            pyautogui_code = actions_to_pyautogui(
                parsed_actions, self.screen_height, self.screen_width
            )

            self.thoughts.append(thought)
            self.actions.append([pyautogui_code])
            self.history_responses.append(response)

            # Reset stuck counter on successful action
            self.consecutive_waits = 0

            return response, [pyautogui_code]

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            self.thoughts.append(response)
            self.actions.append(["FAIL"])
            self.history_responses.append(response)
            return response, ["FAIL"]

    def reset(self, _logger=None, vm_ip: str | None = None):
        """Reset agent state between tasks."""
        global logger
        if _logger is not None:
            logger = _logger

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.consecutive_waits = 0
        self.last_action_type = None

        logger.info("MyUITarsAgent reset")
