from datasets import load_dataset
from transformers import AutoTokenizer
import os
import argparse


parser = argparse.ArgumentParser(description="Tokenize reward model training data")
parser.add_argument("--model_name", type=str, default=None, required=True, help="Model name for tokenizer (e.g., Qwen/Qwen2.5-7B-Instruct)")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train.jsonl from collect_rm_data")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for tokenized data (default: data_dir/tokenized)")
parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for tokenization")
args = parser.parse_args()

# ==== Config ====
MODEL_NAME = args.model_name
prefix = args.data_dir
tokenized_path = args.output_dir if args.output_dir else os.path.join(prefix, "tokenized")

# ==== Validate input ====
train_file = os.path.join(prefix, "train.jsonl")
if not os.path.exists(train_file):
    print(f"❌ Error: train.jsonl not found in {prefix}")
    print(f"   Please run the collect_rm_data pipeline first to generate the training data.")
    print(f"   Expected file: {train_file}")
    exit(1)

print("=" * 60)
print("Tokenize Data for Reward Model Training")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Input: {train_file}")
print(f"Output: {tokenized_path}")
print("=" * 60)

# ==== Create output dir ====
os.makedirs(tokenized_path, exist_ok=True)

# ==== Load raw dataset ====
print("Loading dataset...")
raw_datasets = load_dataset(
    "json",
    data_files={
        "train": train_file,
    },
)

print(f"✅ Loaded dataset: {raw_datasets['train']}")

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples):
    # Convert text to tokens
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )
    # Keep label
    tokenized["labels"] = examples["label"]
    return tokenized

# ==== Map with multiprocessing ====
print("Tokenizing dataset...")
tokenized_datasets = raw_datasets.map(
    preprocess,
    batched=True,
    batch_size=100,
    num_proc=args.num_proc,
    remove_columns=[
        col for col in raw_datasets["train"].column_names if col not in ["text", "label"]
    ],
)

# ==== Save ====
print("Saving tokenized dataset...")
tokenized_datasets.save_to_disk(tokenized_path)
print("=" * 60)
print(f"✅ Tokenized dataset saved to: {tokenized_path}")
print(f"   You can now train the reward model with:")
print(f"   python train_rank.py --model_name {MODEL_NAME} --data_dir {tokenized_path}")
print("=" * 60)