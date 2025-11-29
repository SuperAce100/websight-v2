# Agentnet Dataset Transformation Plan for UI-TARS 1.5 SFT

## Overview

This document outlines the plan for transforming the agentnet dataset into a format suitable for supervised fine-tuning (SFT) of UI-TARS 1.5 using Llama-Factory, with a focus on next-frame action prediction including Chain of Thought (CoT) reasoning.

## Dataset Structure

The agentnet dataset contains desktop agent trajectories with the following structure:

```json
{
  "task_id": "...",
  "instruction": "sort the table in ascending order...",
  "natural_language_task": "Could you help me sort...",
  "actual_task": "Sort a table in WPS Office...",
  "task_completed": false,
  "alignment_score": 7,
  "efficiency_score": 6,
  "task_difficulty": 3,
  "traj": [
    {
      "index": 0,
      "image": "ea83c4aa-a4b1-48af-b439-0de7ee7b8d3f.png",
      "value": {
        "observation": "I'm looking at a WPS Office Excel spreadsheet...",
        "thought": "Since this is the first action...",
        "action": "Click on cell C2...",
        "code": "pyautogui.click(x=0.1632, y=0.2711)",
        "last_step_correct": true,
        "last_step_redundant": false,
        "reflection": "The action has successfully selected..."
      }
    },
    ...
  ]
}
```

## Goal: Next-Frame Action Prediction with CoT

**Objective**: Train the model to predict the next action given:
- The task instruction
- Full history of previous steps (images, observations, thoughts, actions)
- Current state (current image and observation)

**Output**: The model should generate:
1. **Thought** (CoT reasoning): Why this action is appropriate
2. **Action**: Natural language description of what to do
3. **Code**: PyAutoGUI command to execute the action

## Transformation Strategy

### 1. Training Example Generation

For each trajectory of length N, we create **N-1 training examples**:
- Example 0: Predict step 1 given step 0
- Example 1: Predict step 2 given steps 0-1
- Example 2: Predict step 3 given steps 0-2
- ...
- Example N-2: Predict step N-1 given steps 0-(N-2)

This ensures the model learns to:
- Use full context/history
- Reason about next actions (CoT)
- Generate executable code

### 2. Format: Llama-Factory ShareGPT

Each training example follows the ShareGPT format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a desktop automation assistant. Your task is: [task description]..."
    },
    {
      "role": "user",
      "content": "<image>\n**Observation 0:** [observation text]"
    },
    {
      "role": "assistant",
      "content": "**Thought:** [thought]\n**Action:** [action]\n**Code:** pyautogui.click(x, y)"
    },
    {
      "role": "user",
      "content": "<image>\n**Observation 1:** [observation text]"
    },
    {
      "role": "assistant",
      "content": "**Thought:** [thought]\n**Action:** [action]\n**Code:** pyautogui.click(x, y)"
    },
    {
      "role": "user",
      "content": "<image>\n**Current Observation:** [current observation]"
    },
    {
      "role": "assistant",
      "content": "**Thought:** [next thought]\n**Action:** [next action]\n**Code:** pyautogui.click(x, y)"
    }
  ],
  "images": [
    "path/to/image0.png",
    "path/to/image1.png",
    "path/to/image2.png"
  ]
}
```

### 3. Key Design Decisions

#### A. Including CoT in Output
- **Rationale**: The paper indicates CoT was necessary for nontrivial results
- **Implementation**: Include "thought" field in assistant responses
- **Format**: Structured as `**Thought:** [reasoning]` followed by action and code

#### B. Multi-Turn History
- **Rationale**: Next-frame prediction requires full context
- **Implementation**: Include all previous steps as user/assistant pairs
- **Benefit**: Model learns to reason over sequential states

#### C. Coordinate Normalization
- **Current**: Agentnet uses normalized coordinates (0-1 range)
- **Target**: UI-TARS typically uses 1400x800 resolution
- **Implementation**: Convert 0-1 coordinates to absolute pixels (1400x800)
- **Example**: `pyautogui.click(x=0.1632, y=0.2711)` → `pyautogui.click(228, 217)`

#### D. Image Handling
- **Challenge**: Images may not be locally available
- **Options**:
  1. Download images if URLs are provided
  2. Use relative paths if images are in a known directory
  3. Use absolute paths if images are available
- **Implementation**: Support `--base-image-dir` parameter for local images

### 4. Data Splitting

- **Train**: 80% (default)
- **Validation**: 10% (default)
- **Test**: 10% (default)
- **Method**: Random split at trajectory level (not step level) to avoid data leakage

### 5. Filtering

- **Minimum trajectory length**: 2 steps (need at least one prediction)
- **Skip invalid steps**: Missing thought, action, or code
- **Optional**: Filter by `last_step_correct` flag (only use correct steps)

## Usage

### Basic Usage

```bash
python scripts/transform_agentnet_for_training.py \
  --input agentnet/agentnet_win_mac_18k.jsonl \
  --output-dir data \
  --base-image-dir /path/to/images \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

### With Reflection

```bash
python scripts/transform_agentnet_for_training.py \
  --input agentnet/agentnet_win_mac_18k.jsonl \
  --output-dir data \
  --include-reflection
```

### Testing with Limited Examples

```bash
python scripts/transform_agentnet_for_training.py \
  --input agentnet/agentnet_win_mac_18k.jsonl \
  --output-dir data \
  --max-examples-per-trajectory 5
```

## Dataset Registration

After transformation, register the dataset in `configs/dataset_info.json`:

```json
{
  "agentnet_train": {
    "file_name": "agentnet_train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "agentnet_val": {
    "file_name": "agentnet_val.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

## Training Configuration

### Key Considerations for UI-TARS 1.5

1. **Template**: Use the appropriate template for UI-TARS 1.5 (check model card)
2. **Context Length**: Multi-turn conversations can be long; ensure `cutoff_len` is sufficient
3. **Image Processing**: Ensure vision encoder can handle multiple images per example
4. **LoRA Settings**: Consider targeting vision layers if UI-TARS supports it

### Example Training Config

```yaml
model_name_or_path: path/to/ui-tars-1.5
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
dataset: agentnet_train
template: ui_tars  # Check UI-TARS 1.5 template name
cutoff_len: 8192  # Adjust based on trajectory length
output_dir: saves/ui-tars-1.5/agentnet-sft
per_device_train_batch_size: 1  # Multi-turn examples can be large
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 3
```

## Expected Output Statistics

For a dataset with:
- 18,000 trajectories
- Average trajectory length: ~5 steps

Expected output:
- Training examples: ~72,000 (18,000 × 4 average predictions per trajectory)
- Validation examples: ~9,000
- Test examples: ~9,000

## Refinements and Alternatives

### Alternative 1: Simplified Output (No CoT)
If CoT proves too difficult to learn initially:
- Output only: `**Action:** [action]\n**Code:** [code]`
- Can add CoT back later with curriculum learning

### Alternative 2: Separate CoT and Action
Train two models:
1. CoT model: Predicts thought given history
2. Action model: Predicts action+code given history+thought

### Alternative 3: Filtered Training
Only use steps where `last_step_correct: true`:
- Higher quality training data
- Fewer examples but potentially better learning signal

### Alternative 4: Curriculum Learning
1. Phase 1: Short trajectories (2-3 steps)
2. Phase 2: Medium trajectories (4-6 steps)
3. Phase 3: Full trajectories (7+ steps)

## Next Steps

1. **Run transformation**: Execute the script on agentnet dataset
2. **Inspect samples**: Review transformed examples for quality
3. **Handle images**: Download or locate image files
4. **Register dataset**: Add to `dataset_info.json`
5. **Test training**: Run a small training run to verify format
6. **Evaluate**: Check if model learns CoT reasoning patterns

## Questions to Consider

1. **Image availability**: Are images available locally or need to be downloaded?
2. **Coordinate system**: Confirm UI-TARS 1.5's expected coordinate format
3. **Template compatibility**: Verify UI-TARS 1.5 template supports multi-image inputs
4. **Memory constraints**: Long trajectories may require gradient checkpointing
5. **Evaluation metric**: How to measure CoT quality vs. action accuracy?




