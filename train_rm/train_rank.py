#!/usr/bin/env python3
# train_ordinal_rm.py
"""
Ordinal Reward Model training (CORAL-like) for y in {1..5}.
Transforms labels to 3 binary targets: [y>1, y>2, y>3]
Uses HuggingFace Trainer + AutoModelForSequenceClassification(num_labels=3)
"""

import os
import argparse
from datasets import load_from_disk, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
import evaluate
import random
from sklearn.metrics import mean_squared_error
import torch

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(description="Train ordinal reward model")
parser.add_argument("--model_name", type=str, default=None, required=True, help="Base model name (e.g., Qwen/Qwen2.5-7B-Instruct)")
parser.add_argument("--data_dir", type=str, default=None, required=True, help="Directory containing tokenized data from tokenize_data.py")
parser.add_argument("--output_dir", type=str, default="./ordinal_rm_rank_v3_qwen_v6", help="Output directory for trained model")
parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size per device")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--deepspeed", type=str, default="ds_config.json", help="DeepSpeed config file (set to empty string to disable)")
parser.add_argument("--logging_steps", type=int, default=1, help="Logging frequency")
parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
parser.add_argument("--save_steps", type=int, default=300, help="Save frequency")
parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
parser.add_argument("--bf16", action="store_true", help="Use bfloat16 training")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# =====================
# Validate inputs
# =====================
if not os.path.exists(args.data_dir):
    print(f"❌ Error: Data directory not found: {args.data_dir}")
    print(f"   Please run tokenize_data.py first to generate tokenized data.")
    exit(1)

print("=" * 70)
print("Train Ordinal Reward Model")
print("=" * 70)
print(f"Base Model: {args.model_name}")
print(f"Data Directory: {args.data_dir}")
print(f"Output Directory: {args.output_dir}")
print(f"Batch Size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} grad accum")
print(f"Learning Rate: {args.lr}")
print(f"Epochs: {args.num_train_epochs}")
print(f"BF16: {args.bf16}")
print("=" * 70)



def set_seed(seed: int = 42):
    """
    设置全局随机种子以保证实验可复现性
    """
    # 1. Python 原生 random
    random.seed(seed)
    
    # 2. 环境变量 (影响 Hash 行为，比如字典顺序)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch (CPU)
    torch.manual_seed(seed)
    
    # 5. PyTorch (GPU / CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果是多卡训练，这一步很重要
        
    print(f"Random seed set to: {seed}")

# ----------------------------
# Utils: ordinal encoding / decoding
# ----------------------------
def ordinal_encode_label(y):
    """
    Convert scalar y in {1,2,3,4,5} to length-4 float vector:
    [y>1, y>2, y>3, y>4]
    """
    return [float(y > 1), float(y > 2), float(y > 3)]

def decode_ordinal_logits_to_expectation(logits_np):
    """
    logits_np: ndarray shape [B,4]
    return: expected value E[y] (float) per example and class probabilities shape [B,5]
    """
    # sigmoid to get P(y > k)
    p = 1.0 / (1.0 + np.exp(-logits_np))  # [B,3]
    p1 = 1.0 - p[:, 0]
    p2 = p[:, 0] - p[:, 1]
    p3 = p[:, 1] - p[:, 2]
    p4 = p[:, 2]
    # p4 = p[:, 2] - p[:, 3]
    # p5 = p[:, 3]
    # probs = np.stack([p1, p2, p3, p4, p5], axis=1)  # [B,5]
    probs = np.stack([p1, p2, p3, p4], axis=1)  # [B,4]

    classes = np.arange(1, 5)
    exp = (probs * classes[None, :]).sum(axis=1)
    return exp, probs

# ----------------------------
# Load dataset
# ----------------------------
set_seed(args.seed)
print(f"Loading dataset from: {args.data_dir}")
dataset = load_from_disk(args.data_dir)
print(f"✅ Dataset loaded: {dataset}")

# ensure labels column exists and is int/float
if "labels" not in dataset["train"].column_names:
    print(f"❌ Error: Dataset must contain 'labels' column with values in {{1,2,3,4}}")
    print(f"   Available columns: {dataset['train'].column_names}")
    exit(1)

# cast labels to float (scalar) first if needed
dataset = dataset.cast_column("labels", Value("float32"))

# map to ordinal binary vector
def map_to_ordinal(example):
    # if labels already vector, skip (but we expect scalar)
    y = example["labels"]
    # handle potential NaNs or invalids
    try:
        yv = float(y)
    except Exception:
        yv = 3.0
    if yv < 1:
        yv = 1.0
    if yv > 5:
        yv = 5.0
    example["labels"] = ordinal_encode_label(int(round(yv)))
    return example

print("Encoding labels into ordinal (4 binary tasks)...")
dataset = dataset.map(map_to_ordinal, batched=False)
# set format to torch tensors (input_ids, attention_mask, labels)
# assume dataset already tokenized and contains input_ids, attention_mask
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ----------------------------
# Tokenizer & Model
# ----------------------------
print(f"Loading tokenizer and model: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# num_labels = 3 for ordinal tasks
num_labels = 3
print(f"Loading model with {num_labels} ordinal labels...")
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
# ensure multi-label problem type (so trainer/model uses BCEWithLogitsLoss)
model.config.problem_type = "multi_label_classification"
model.config.pad_token_id = tokenizer.pad_token_id
print("✅ Model loaded successfully")

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (B,4), labels: (B,4)
    # convert logits -> expectation E[y]
    logits_np = logits
    if isinstance(logits, tuple) or hasattr(logits, "logits"):
        logits_np = logits[0]
    logits_np = np.asarray(logits_np)
    labels_np = np.asarray(labels)

    # predicted expected y
    preds_exp, probs = decode_ordinal_logits_to_expectation(logits_np)

    # recover ground-truth scalar y from ordinal binary vector
    gt = 1 + (labels_np > 0.5).sum(axis=1)  # shape [B], values in 1..5

    mse = mean_squared_error(gt, preds_exp)
    rmse = float(np.sqrt(mse))

    # we also report classification accuracy for ordinal >k tasks (optional)
    # compute binary acc per threshold
    pred_p = 1.0 / (1.0 + np.exp(-logits_np))
    bin_pred = (pred_p > 0.5).astype(float)
    bin_gt = labels_np.astype(float)
    bin_acc = (bin_pred == bin_gt).mean()

    return {"mse": mse, "rmse": rmse, "ordinal_bin_acc": float(bin_acc)}

# ----------------------------
# TrainingArguments + Trainer
# ----------------------------
# Handle DeepSpeed config
deepspeed_config = None
if args.deepspeed and os.path.exists(args.deepspeed):
    deepspeed_config = args.deepspeed
    print(f"Using DeepSpeed config: {args.deepspeed}")
elif args.deepspeed:
    print(f"⚠️ Warning: DeepSpeed config file not found: {args.deepspeed}")
    print(f"   Training will proceed without DeepSpeed")

os.makedirs(args.output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    # eval_strategy="steps",
    # eval_steps=args.eval_steps,
    # save_strategy="steps",
    # save_steps=args.save_steps,
    learning_rate=args.lr,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    warmup_ratio=0.05,
    logging_dir=os.path.join(args.output_dir, "logs"),
    logging_steps=args.logging_steps,
    bf16=args.bf16,
    max_grad_norm=1.0,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    deepspeed=deepspeed_config,
    report_to="wandb",
    # save_total_limit=10,
    # load_best_model_at_end=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    torch_compile=True,
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=None,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ----------------------------
# Train
# ----------------------------
print("\n" + "=" * 70)
print("Starting training...")
print("=" * 70)
trainer.train()

# ----------------------------
# Save model & tokenizer
# ----------------------------
print("\n" + "=" * 70)
print(f"Saving final model to: {args.output_dir}")
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("✅ Model saved successfully!")
print("=" * 70)
print("\nTraining completed! You can use the trained model for inference.")
