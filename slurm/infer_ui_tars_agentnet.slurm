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

image_path="/hai/users/a/s/asanshay/websight-v2/sample.png"
prompt="Click the Add New button"
model_id="Asanshay/websight-2-7B-kto"

print(f"Loading model: {model_id}")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

image = Image.open(image_path).convert("RGB")

system_prompt = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: # Put PyAutoGUI Code Here
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use ENGLISH in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Then put the action itself in the `Action` part.

YOU MUST INCLUDE THE ACTION IN THE ACTION PART
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
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

result = processor.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
print("\nRaw model output:")
print(result)
PY

echo ""
echo "End : $(date)"

