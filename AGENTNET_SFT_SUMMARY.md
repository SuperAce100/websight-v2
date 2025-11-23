# Agentnet SFT Transformation Summary

## Quick Answer

**Goal**: Transform agentnet dataset for next-frame action prediction SFT on UI-TARS 1.5 with Chain of Thought (CoT) reasoning.

**Approach**: 
- For each trajectory step `i`, create a training example that predicts step `i+1`
- Input: Task instruction + full history (all previous images, observations, thoughts, actions)
- Output: Next thought (CoT) + action + pyautogui code

## Key Design Decisions

### 1. Including CoT in Output ✅
**Why**: The paper indicates CoT was necessary for nontrivial results.

**How**: Include the "thought" field in the assistant's response:
```
**Thought:** [reasoning about why this action is appropriate]
**Action:** [natural language description]
**Code:** pyautogui.click(x, y)
```

### 2. Multi-Turn History ✅
**Why**: Next-frame prediction requires full context of what happened before.

**How**: Format as a conversation where:
- Each previous step is a user/assistant pair (image + observation → thought + action + code)
- The final turn is: current state → predict next action

### 3. Coordinate Normalization ✅
**Why**: Agentnet uses normalized (0-1) coordinates, but UI-TARS typically uses 1400x800 pixels.

**How**: Automatically convert `pyautogui.click(x=0.1632, y=0.2711)` → `pyautogui.click(228, 217)`

## Transformation Script

**Location**: `scripts/transform_agentnet_for_training.py`

**Usage**:
```bash
python scripts/transform_agentnet_for_training.py \
  --input agentnet/agentnet_win_mac_18k.jsonl \
  --output-dir data \
  --base-image-dir /path/to/images  # If images are local
```

**Output**: Creates `agentnet_train.jsonl`, `agentnet_val.jsonl`, `agentnet_test.jsonl`

## Output Format

Each training example follows Llama-Factory's ShareGPT format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a desktop automation assistant. Your task is: [task]..."
    },
    {
      "role": "user",
      "content": "<image>\n**Observation 0:** [observation]"
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
  "images": ["image0.png", "image1.png", "image2.png"]
}
```

## Dataset Registration

Already added to `configs/dataset_info.json`:
- `agentnet_train`
- `agentnet_val`
- `agentnet_test`

## Next Steps

1. **Handle Images**: 
   - If images are URLs, download them
   - If images are local, set `--base-image-dir` to their location
   - If images aren't available, the script will still work (just won't include image paths)

2. **Run Transformation**:
   ```bash
   python scripts/transform_agentnet_for_training.py \
     --input agentnet/agentnet_win_mac_18k.jsonl \
     --output-dir data
   ```

3. **Inspect Output**:
   ```bash
   head -n 1 data/agentnet_train.jsonl | python3 -m json.tool
   ```

4. **Train with Llama-Factory**:
   - Use dataset: `agentnet_train`
   - Ensure template matches UI-TARS 1.5 requirements
   - Set appropriate `cutoff_len` for long trajectories

## Important Notes

- **Image Paths**: The script handles missing images gracefully. If images aren't available, the examples will still be created but without image references.
- **Trajectory Length**: Each trajectory of length N creates N-1 training examples (one for each step where we can predict the next).
- **Data Splitting**: Split happens at trajectory level (not step level) to avoid data leakage.
- **CoT Format**: The structured format (`**Thought:**`, `**Action:**`, `**Code:**`) makes it easy for the model to learn the pattern.

## Refinements (If Needed)

If initial training shows issues:

1. **Simplify Output**: Remove CoT initially, add back later
2. **Filter Quality**: Only use steps where `last_step_correct: true`
3. **Curriculum Learning**: Start with short trajectories, gradually increase length
4. **Separate Models**: Train CoT model separately from action model

See `AGENTNET_TRANSFORMATION_PLAN.md` for detailed documentation.




