# Train WildReward Model

This directory contains scripts for training an ordinal reward model using data collected from the `collect_rm_data` pipeline.

## Overview

The training process consists of two steps:
1. **Tokenize** the raw training data into a format suitable for model training
2. **Train** the reward model using ordinal regression (CORAL-like method)

## Prerequisites

1. **Data Preparation**: First run the `collect_rm_data` pipeline to generate the training data:
   ```bash
   cd ../collect_rm_data
   # Follow the README there to generate train.jsonl
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r ../collect_rm_data/requirements.txt
   pip install deepspeed accelerate
   ```

3. **GPU Requirements**: Multi-GPU training is recommended (8 GPUs configured by default)

## Quick Start

The easiest way to train is using the provided `run.sh` script:

```bash
./run.sh <MODEL_NAME> <DATA_DIR> <OUTPUT_DIR>
```

### Example

```bash
./run.sh Qwen/Qwen2.5-7B-Instruct \
    ../collect_rm_data/output \
    ./my_reward_model
```

This will:
1. Tokenize the data from `../collect_rm_data/output/train.jsonl`
2. Train the reward model
3. Save the final model to `./my_reward_model`

## Individual Steps

If you prefer to run each step separately:

### Step 1: Tokenize Data

Convert raw JSONL data into tokenized format:

```bash
python tokenize_data.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --data_dir ../collect_rm_data/output \
    --output_dir ./tokenized_data \
    --max_length 4096 \
    --num_proc 4
```

**Arguments:**
- `--model_name`: HuggingFace model name for the tokenizer (required)
- `--data_dir`: Directory containing `train.jsonl` from `collect_rm_data` (required)
- `--output_dir`: Where to save tokenized data (default: `data_dir/tokenized`)
- `--max_length`: Maximum sequence length (default: 4096)
- `--num_proc`: Number of processes for tokenization (default: 4)

**Output:** Creates a tokenized dataset that can be loaded with `load_from_disk()`

### Step 2: Train Model

Train the ordinal reward model:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --num_processes 8 \
    --main_process_port 29502 \
    train_rank.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --data_dir ./tokenized_data \
    --output_dir ./my_reward_model \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr 1e-5 \
    --num_train_epochs 1 \
    --bf16
```

**Key Arguments:**
- `--model_name`: Base model to fine-tune (required)
- `--data_dir`: Directory containing tokenized data from Step 1 (required)
- `--output_dir`: Where to save the trained model (default: `./ordinal_rm_rank_v3_qwen_v6`)
- `--per_device_train_batch_size`: Batch size per GPU (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 16)
- `--lr`: Learning rate (default: 1e-5)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--bf16`: Use bfloat16 training (recommended)
- `--deepspeed`: DeepSpeed config file (default: `ds_config.json`, use empty string to disable)

**Output:** Trained model saved to `--output_dir`

## Model Architecture

The reward model uses **ordinal regression** with a CORAL-like approach:

- **Labels**: Discrete ratings in {1, 2, 3, 4}
- **Transformation**: Converts labels to 3 binary targets `[y>1, y>2, y>3]`
- **Model**: `AutoModelForSequenceClassification` with `num_labels=3`
- **Loss**: Binary cross-entropy (BCEWithLogitsLoss) for multi-label classification
- **Prediction**: Converts logits to expected value E[y] using probability thresholds

This approach is more effective than standard regression for ordinal data with discrete levels.

## Configuration

### DeepSpeed

The `ds_config.json` file contains DeepSpeed configuration for distributed training. Adjust based on your GPU memory:

- `train_micro_batch_size_per_gpu`: Per-GPU batch size
- `gradient_accumulation_steps`: Gradient accumulation
- `fp16`/`bf16`: Mixed precision training

### Hyperparameters

Common hyperparameters to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 1e-5 | Learning rate |
| `--num_train_epochs` | 1 | Number of epochs |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--gradient_accumulation_steps` | 16 | Gradient accumulation |
| `--warmup_ratio` | 0.1 | Warmup ratio |

## Monitoring

Training logs are reported to Weights & Biases (wandb) by default. To monitor training:

```bash
# Make sure wandb is installed and logged in
pip install wandb
wandb login
```

## Using the Trained Model

After training, use your model for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./my_reward_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Prepare input
text = "Your conversation text here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: [1, 3]

# Convert to rating
# The model outputs 3 logits for [y>1, y>2, y>3]
# Convert to expected rating using the ordinal decoding logic
```

## Troubleshooting

### "train.jsonl not found"
Make sure you've completed the `collect_rm_data` pipeline first:
```bash
cd ../collect_rm_data
bash run_pipeline.sh
```

### CUDA Out of Memory
- Reduce `--per_device_train_batch_size`
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Use gradient checkpointing (enabled by default)

### DeepSpeed Issues
- Verify `ds_config.json` exists in the current directory
- Or disable DeepSpeed: `--deepspeed ""`

## File Structure

```
train_rm/
├── README.md              # This file
├── run.sh                 # Full training pipeline script
├── tokenize_data.py       # Step 1: Tokenize raw data
├── train_rank.py          # Step 2: Train reward model
└── ds_config.json         # DeepSpeed configuration
```

## References

- **collect_rm_data**: Data collection pipeline (see `../collect_rm_data/README.md`)
- **CORAL**: COnsistent RAnk Logits for ordinal regression
