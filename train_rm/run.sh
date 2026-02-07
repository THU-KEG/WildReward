#!/bin/bash
# run.sh - Full training pipeline for reward model
# Usage: ./run.sh <MODEL_NAME> <DATA_DIR> <OUTPUT_DIR>
#
# Arguments:
#   MODEL_NAME  - HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct)
#   DATA_DIR    - Directory containing train.jsonl from collect_rm_data
#   OUTPUT_DIR  - Where to save the trained model
#
# Example:
#   ./run.sh Qwen/Qwen2.5-7B-Instruct \
#       ../collect_rm_data/output \
#       ./my_reward_model

set -e  # Exit on error

MODEL_NAME=$1
DATA_DIR=$2
OUTPUT_DIR=$3

# Validate arguments
if [ -z "$MODEL_NAME" ] || [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Error: Missing arguments"
    echo "Usage: $0 <MODEL_NAME> <DATA_DIR> <OUTPUT_DIR>"
    echo ""
    echo "Example:"
    echo "  $0 Qwen/Qwen2.5-7B-Instruct ../collect_rm_data/output ./my_reward_model"
    exit 1
fi

# Validate data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Validate train.jsonl exists
TRAIN_FILE="$DATA_DIR/train.jsonl"
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: train.jsonl not found in $DATA_DIR"
    echo "   Please run the collect_rm_data pipeline first."
    exit 1
fi

echo "=========================================="
echo "Reward Model Training Pipeline"
echo "=========================================="
echo "Model:     $MODEL_NAME"
echo "Data:      $DATA_DIR"
echo "Output:    $OUTPUT_DIR"
echo "=========================================="

# Step 1: Tokenize data
echo ""
echo "Step 1/2: Tokenizing data..."
python tokenize_data.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${DATA_DIR}"

# Step 2: Train model
echo ""
echo "Step 2/2: Training reward model..."
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --num_processes 8 \
    --main_process_port 29502 \
    train_rank.py \
    --data_dir "${DATA_DIR}/tokenized" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --bf16

echo ""
echo "=========================================="
echo "✅ Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="

