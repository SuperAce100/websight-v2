#!/bin/bash
#SBATCH --job-name=infer_ui_tars
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --output=logs/infer_ui_tars_%j.out
#SBATCH --error=logs/infer_ui_tars_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Example inference on the merged UI-TARS AgentNet model.
# Expects a local image and runs a single prompt using transformers.

set -euo pipefail

WORKSPACE_DIR="/hai/users/a/s/asanshay/websight-v2"
cd "${WORKSPACE_DIR}"



echo "Workspace : ${WORKSPACE_DIR}"
echo "Start     : $(date)"
echo ""


# Caches to scratch
SCRATCH_ROOT="/hai/scratch/asanshay/websight-v2"
CACHE_ROOT="${SCRATCH_ROOT}/.cache"
HF_HOME="${CACHE_ROOT}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE TRITON_CACHE_DIR
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRITON_CACHE_DIR}"

# Env
if [ ! -d "venv" ]; then
    echo "✗ Error: venv not found in ${WORKSPACE_DIR}"
    exit 1
fi
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python - <<'PY'
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# Reduce allocator fragmentation on large generations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

image_path="/hai/users/a/s/asanshay/websight-v2/sample.png"
prompt="Click the Add New button"
model_id="Asanshay/websight-2-7B-kto"
max_new_tokens=128

print(f"Loading model: {model_id}")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

image = Image.open(image_path).convert("RGB")

system_prompt = """
You are a GUI agent. You are given a task and a screenshot of the screen. You 
need to perform a series of pyautogui actions to complete the task.

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

Your thought should be extremely brief, less than 50 words.

**Code:** Finally, output the action as a PyAutoGUI command (e.g. 
pyautogui.click(x, y), etc.) or the following functions:
- {"name": "computer.triple_click", "description": "Triple click on the screen",
"parameters": {"type": "object", "properties": {"x": {"type": "number", "description": 
"The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y 
coordinate of the triple click"}}, "required": ["x", "y"]}}
- {"name": "computer.terminate", "description": "Terminate the current task and report 
its completion status", "parameters": {"type": "object", "properties": {"status": {"type":
"string", "enum": ["success", "failure"], "description": "The status of the task"}}, 
"required": ["status"]}}

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
            {"type": "text", "text": f"This is the state of my computer right now. Create a new project."},
        ],
    },
]
chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=chat_text,
    images=[image],
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(**inputs,  max_new_tokens=max_new_tokens, do_sample=False)

result = processor.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
print("\nRaw model output:")
print(result)
PY

echo ""
echo "End : $(date)"

