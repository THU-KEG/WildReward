#!/bin/bash
# ==============================================================================
# One-Click Pipeline Script for Reward Model Data Processing
# ==============================================================================
# This script automates the entire data processing pipeline for collecting
# reward model data from WildChat conversations.
#
# Pipeline Steps (executed in order):
#   00. Preprocess raw WildChat parquet data to JSONL format
#   01. Generate preference classification prompts from conversations
#   02. Call model API to generate responses for each prompt
#   03. Filter and parse model outputs to extract labels
#   04. Merge conversation turns into conversation format
#   05. Implicit feedback mining with topic-aware hindsight
#   06. Generate refusal validation prompts for negative samples
#   07. Merge refusal validation results into final dataset
#   08. Final filtering and train/test split
#
# Usage:
#   ./run_pipeline.sh --input_dir <path> --model_name <name>
#
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# Configuration - User should modify these variables
# ==============================================================================

# ====== Environment Variables for Model API ======
# Set these before running or modify here
export BASE_URL="${BASE_URL:-http://localhost:8001/v1}"
export API_KEY="${API_KEY:-NA}"

# ====== Input/Output Directory Configuration ======
WILDCHAT_DIR="${WILDCHAT_DIR:-./wildchat_data}"           # Raw WildChat parquet files
OUTPUT_BASE="${OUTPUT_BASE:-./pipeline_output}"            # Base output directory
MODEL_NAME="${MODEL_NAME:-gpt-oss-120b}"                   # Model name for generation

# ====== Model Generation Parameters ======
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.5}"
NUM_THREADS="${NUM_THREADS:-512}"

# ====== CUDA Device ======
CUDA_DEVICE="${CUDA_DEVICE:-0}"                            # GPU device ID for embedding model

# ====== Intermediate Directories (auto-generated) ======
PROCESSED_DIR="$OUTPUT_BASE/processed_data"                # Step 00 output
PROMPT_DIR="$OUTPUT_BASE/generated_prompts"               # Step 01 output
RESPONSE_DIR="$OUTPUT_BASE/$MODEL_NAME"                    # Step 02 output
FILTERED_DIR="$OUTPUT_BASE/${MODEL_NAME}-filtered"         # Step 03 output
MERGED_DIR="$OUTPUT_BASE/data"                             # Step 04 output
HINDSIGHT_DIR="$OUTPUT_BASE/hindsight"                     # Step 05 output
REFUSAL_PROMPT_DIR="$OUTPUT_BASE/refusal_prompts"          # Step 06 output
REFUSAL_OUTPUT_DIR="$OUTPUT_BASE/refusal_output"           # Step 06 generation output
FINAL_DIR="$OUTPUT_BASE/final"                             # Step 07 output

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            WILDCHAT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --base_url)
            export BASE_URL="$2"
            shift 2
            ;;
        --api_key)
            export API_KEY="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --cuda_device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input_dir PATH       Directory containing WildChat parquet files"
            echo "  --model_name NAME      Model name for generation (default: gpt-oss-120b)"
            echo "  --base_url URL         Base URL for model API"
            echo "  --api_key KEY          API key for model API"
            echo "  --output_dir PATH      Base output directory (default: ./pipeline_output)"
            echo "  --cuda_device ID       CUDA device ID (default: 0)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  BASE_URL               Base URL for model API"
            echo "  API_KEY                API key for model API"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Print Configuration
# ==============================================================================

echo "=========================================================================="
echo "Reward Model Data Processing Pipeline"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Input Directory:      $WILDCHAT_DIR"
echo "  Output Base:          $OUTPUT_BASE"
echo "  Model Name:           $MODEL_NAME"
echo "  Base URL:             $BASE_URL"
echo "  CUDA Device:          $CUDA_DEVICE"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 00: Preprocess WildChat Data
# ==============================================================================

echo "====== Step 00: Preprocessing WildChat Data ======"
echo "Input: $WILDCHAT_DIR"
echo "Output: $PROCESSED_DIR"
echo ""

mkdir -p "$PROCESSED_DIR"

python3 00.preprocess_wildchat.py \
    --input_dir "$WILDCHAT_DIR" \
    --output_dir "$PROCESSED_DIR"

echo "✅ Step 00 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 01: Generate Preference Prompts
# ==============================================================================

echo "====== Step 01: Generating Preference Prompts ======"
echo "Input: $PROCESSED_DIR"
echo "Output: $PROMPT_DIR"
echo ""

mkdir -p "$PROMPT_DIR"

# Process all JSONL files from step 00
for file in "$PROCESSED_DIR"/*.jsonl; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        base="${filename%.*}"

        output_file="$PROMPT_DIR/${base}_prompts.jsonl"

        echo ">>> Processing: $filename"

        python3 01.gen_action.py \
            --input "$file" \
            --output "$output_file"

        echo "✅ Done: $output_file"
        echo "----------------------------------------"
    fi
done

echo "✅ Step 01 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 02: Generate Model Responses
# ==============================================================================

echo "====== Step 02: Generating Model Responses ======"
echo "Input: $PROMPT_DIR"
echo "Output: $RESPONSE_DIR"
echo ""

mkdir -p "$RESPONSE_DIR"

# Process all prompt files from step 01
for prompt_file in "$PROMPT_DIR"/*_prompts.jsonl; do
    if [ -f "$prompt_file" ]; then
        filename=$(basename -- "$prompt_file")
        base="${filename%_prompts.jsonl}"

        save_name="${base}_response.jsonl"

        echo ">>> Generating model response for: $filename"

        python3 02.generate.py \
            --input_file "$prompt_file" \
            --save_dir "$RESPONSE_DIR" \
            --save_name "$save_name" \
            --model_name_or_path "$MODEL_NAME" \
            --max_tokens "$MAX_TOKENS" \
            --temperature "$TEMPERATURE" \
            --num_threads "$NUM_THREADS" \
            --api_model \
            --n 1

        echo "✅ Done: $save_name"
        echo "----------------------------------------"
    fi
done

echo "✅ Step 02 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 03: Filter and Parse Model Outputs
# ==============================================================================

echo "====== Step 03: Filtering and Parsing Model Outputs ======"
echo "Input: $RESPONSE_DIR"
echo "Output: $FILTERED_DIR"
echo ""

mkdir -p "$FILTERED_DIR"

# Process all response files from step 02
for file in "$RESPONSE_DIR"/*_response.jsonl; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        base="${filename%_response.jsonl}"

        output_file="$FILTERED_DIR/${base}.jsonl"

        echo ">>> Processing: $filename"

        python3 03.gen_train.py \
            --input "$file" \
            --output "$output_file"

        echo "✅ Done: $output_file"
        echo "----------------------------------------"
    fi
done

# Merge all filtered files into one
echo ">>> Merging all filtered files..."
MERGED_FILTERED="$FILTERED_DIR/merged.jsonl"
cat "$FILTERED_DIR"/*.jsonl | grep -v "^$" > "$MERGED_FILTERED"
echo "✅ Merged filtered file: $MERGED_FILTERED"

echo "✅ Step 03 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 04: Merge Conversation Turns
# ==============================================================================

echo "====== Step 04: Merging Conversation Turns ======"
echo "Input: $MERGED_FILTERED"
echo "Output: $MERGED_DIR/merge_convs.jsonl"
echo ""

mkdir -p "$MERGED_DIR"

python3 04.merge.py \
    --input "$MERGED_FILTERED" \
    --output "$MERGED_DIR/merge_convs.jsonl"

echo "✅ Step 04 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 05: Implicit Feedback Mining (Hindsight)
# ==============================================================================

echo "====== Step 05: Implicit Feedback Mining ======"
echo "Input: $MERGED_DIR/merge_convs.jsonl"
echo "Output: $HINDSIGHT_DIR/added_data.jsonl"
echo ""

mkdir -p "$HINDSIGHT_DIR"

python3 05.implicit_feedback_mining.py \
    --input "$MERGED_DIR/merge_convs.jsonl" \
    --output "$HINDSIGHT_DIR/added_data.jsonl" \
    --cuda_device "$CUDA_DEVICE"

echo "✅ Step 05 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 06: Generate Refusal Validation Prompts
# ==============================================================================

echo "====== Step 06: Generating Refusal Validation Prompts ======"
echo "Input: $MERGED_FILTERED"
echo "Output: $REFUSAL_PROMPT_DIR/safe_judge_prompts.jsonl"
echo ""

mkdir -p "$REFUSAL_PROMPT_DIR"
mkdir -p "$REFUSAL_OUTPUT_DIR"

python3 06.gen_refusal_validation.py \
    --input "$MERGED_FILTERED" \
    --output "$REFUSAL_PROMPT_DIR/safe_judge_prompts.jsonl"

echo "✅ Refusal validation prompts generated"
echo ""

# Generate model responses for refusal validation
echo ">>> Generating model responses for refusal validation..."
python3 02.generate.py \
    --input_file "$REFUSAL_PROMPT_DIR/safe_judge_prompts.jsonl" \
    --save_dir "$REFUSAL_OUTPUT_DIR" \
    --save_name "safe_judge_annotated.jsonl" \
    --model_name_or_path "$MODEL_NAME" \
    --max_tokens 256 \
    --temperature 0.5 \
    --num_threads "$NUM_THREADS" \
    --api_model \
    --n 1

echo "✅ Step 06 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 07: Merge Refusal Validation Results
# ==============================================================================

echo "====== Step 07: Merging Refusal Validation Results ======"
echo "Input: $MERGED_FILTERED"
echo "Reann: $REFUSAL_OUTPUT_DIR/safe_judge_annotated.jsonl"
echo "Output: $FINAL_DIR/final.jsonl"
echo ""

mkdir -p "$FINAL_DIR"

python3 07.merge_refusal_validation.py \
    --input "$MERGED_FILTERED" \
    --reann "$REFUSAL_OUTPUT_DIR/safe_judge_annotated.jsonl" \
    --output "$FINAL_DIR/final.jsonl"

echo "✅ Step 07 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Step 08: Final Filtering and Train/Test Split
# ==============================================================================

echo "====== Step 08: Final Filtering and Train/Test Split ======"
echo "Input1: $FINAL_DIR/final.jsonl"
echo "Input2: $HINDSIGHT_DIR/added_data.jsonl"
echo "Train Output: $FINAL_DIR/train.jsonl"
echo "Test Output: $FINAL_DIR/test.jsonl"
echo ""

python3 08.split.py \
    --input1 "$FINAL_DIR/final.jsonl" \
    --input2 "$HINDSIGHT_DIR/added_data.jsonl" \
    --train_output "$FINAL_DIR/train.jsonl" \
    --test_output "$FINAL_DIR/test.jsonl" \
    --test_count 5000

echo "✅ Step 08 completed"
echo ""
echo "=========================================================================="
echo ""

# ==============================================================================
# Pipeline Complete
# ==============================================================================

echo "=========================================================================="
echo "🎉 Pipeline Complete!"
echo "=========================================================================="
echo ""
echo "Final Output Files:"
echo "  Train Set:  $FINAL_DIR/train.jsonl"
echo "  Test Set:   $FINAL_DIR/test.jsonl"
echo ""
echo "Intermediate files are preserved in: $OUTPUT_BASE"
echo ""
echo "=========================================================================="
