# WebSight V2 - GUI Agent Training

Training pipeline for UI-TARS 1.5 vision-language model using AgentNet dataset.

## Quick Start

### 1. Data Preparation

Transform AgentNet data with 70/30 split (70% for SFT, 30% for KTO):

```bash
python scripts/agentnet_scripts/transform_agentnet_for_training.py \
  --input /path/to/agentnet_all.jsonl \
  --output-dir data \
  --base-image-dir /path/to/images \
  --kto-ratio 0.3 \
  --save-kto-split
```

See: [Data Split Documentation](docs/DATA_SPLIT_70_30.md)

### 2. SFT Training

```bash
sbatch --account=ingrai slurm/train_ui_tars_agentnet.slurm
```

### 3. KTO Training

Prepare KTO data:

```bash
bash scripts/data_prep/prepare_kto_data.sh
```

Train with KTO:

```bash
sbatch --account=ingrai slurm/train_agentnet_kto.slurm
```

See: [KTO Training Setup](docs/KTO_TRAINING_SETUP.md)

### INFERENCE 🧠

```bash
python scripts/infer_ui_tars_agentnet.py
```

## Documentation

### Training

- [KTO Training Setup](docs/KTO_TRAINING_SETUP.md) - Complete KTO training guide
- [Data Split 70/30](docs/DATA_SPLIT_70_30.md) - How data is split between SFT and KTO
- [Plot Training Curves](docs/PLOT_TRAINING_CURVES.md) - Visualize training progress

### Model Operations

- [Merge Model](docs/MERGE_MODEL_README.md) - Merge LoRA adapters with base model
- [Merge Model Usage](docs/MERGE_MODEL_USAGE.md) - Quick merge guide
- [Checkpoint to Inference](docs/checkpoint_to_inference.md) - Prepare models for inference

## Project Structure

```
websight-v2/
├── configs/              # Training configurations
│   ├── dataset_info.json
│   ├── train_agentnet_lora.yaml
│   └── train_agentnet_kto.yaml
├── data/                 # Processed datasets
│   ├── agentnet_train.jsonl
│   ├── agentnet_val.jsonl
│   ├── agentnet_test.jsonl
│   ├── agentnet_kto_split.jsonl
│   └── agentnet_kto.jsonl
├── scripts/
│   ├── agentnet_scripts/
│   │   ├── transform_agentnet_for_training.py  # SFT transformation
│   │   └── transform_agentnet_for_kto.py       # KTO transformation
│   ├── data_prep/        # Data preparation scripts
│   ├── evaluation/       # Evaluation scripts
│   ├── model_ops/        # Model operation scripts
│   └── training/         # Training utilities
├── slurm/                # SLURM job scripts
└── docs/                 # Documentation
```

## Key Features

### Data Split Strategy

- **70% for SFT**: Split into train (80%), val (10%), test (10%)
- **30% for KTO**: Reserved for preference-based optimization
- No overlap between SFT and KTO data

### Training Pipeline

1. **SFT (Supervised Fine-Tuning)**: Train on next-frame action prediction
2. **KTO (Preference Optimization)**: Refine using task completion signals

### Model Architecture

- Base model: UI-TARS 1.5 (7B parameters)
- Fine-tuning: LoRA adapters
- Task: Desktop automation with PyAutoGUI code generation

## Requirements

See `requirements-training.txt` for dependencies.

## Citation

If you use this code or data, please cite:

- AgentNet dataset
- UI-TARS model
- LLaMA-Factory framework
