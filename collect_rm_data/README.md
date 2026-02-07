# Wild Reward Model Data Collection Pipeline

This repository contains a complete pipeline for processing WildChat conversations to collect reward model training data through preference labeling and implicit feedback mining.

## Overview

The pipeline processes raw WildChat conversation data through multiple stages:

1. **Preprocessing**: Convert WildChat parquet files to JSONL format
2. **Prompt Generation**: Generate preference classification prompts
3. **Model Inference**: Call LLM API to generate responses
4. **Label Parsing**: Extract preference labels from model outputs
5. **Conversation Merging**: Merge turns into conversation format
6. **Hindsight Mining**: Apply topic-aware implicit feedback mining
7. **Refusal Validation**: Validate and filter refusal responses
8. **Final Split**: Create train/test splits

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for embedding model in step 05)
- 32GB+ RAM
- ~500GB disk space (depending on dataset size)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd collect_rm_data

# Install Python dependencies
pip install -r requirements.txt
```

Create `requirements.txt` if not exists:

```txt
pandas
pyarrow
tqdm
sentence-transformers
datasets
torch
numpy
```

## Step 1: Download WildChat Dataset

Download the WildChat-4.8M dataset from HuggingFace:

```bash
# Using HuggingFace CLI (recommended)
pip install huggingface_hub
huggingface-cli download allenai/WildChat-4.8M --local-dir ./wildchat_data --repo-type dataset
```

Or using Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="allenai/WildChat-4.8M",
    local_dir="./wildchat_data",
    repo_type="dataset"
)
```

Or manually download from: https://huggingface.co/datasets/allenai/WildChat-4.8M

**Note**: The dataset is large (~50GB compressed). Download may take time depending on your connection.

## Step 2: Deploy LLM Service

You have two options: deploy a local LLM service or use a remote API.

### Option A: Deploy Local LLM Service (vLLM)

vLLM is recommended for high-throughput inference:

```bash
# Install vLLM
pip install vllm

# Start vLLM server with your model
python -m vllm.entrypoints.openai.api_server \
    --model <model-path> \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --max-model-len 8192
```

Replace `<model-path>` with your model, e.g.,:
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### Option B: Use Remote API

If you have access to a remote LLM API (OpenAI, Anthropic, custom endpoint):

```bash
# Set your API credentials
export BASE_URL="https://your-api-endpoint.com/v1"
export API_KEY="your-api-key"
```

## Step 3: Configure Environment Variables

Set required environment variables:

```bash
# LLM API Configuration
export BASE_URL="http://localhost:8001/v1"  # Your API endpoint
export API_KEY="NA"                          # Your API key

# Optional: CUDA device for embedding model
export CUDA_DEVICE="0"

# Optional: Model name for tracking
export MODEL_NAME="gpt-oss-120b"
```

## Step 4: Run the Pipeline

### Quick Start (One Command)

```bash
./run_pipeline.sh --input_dir ./wildchat_data --model_name gpt-oss-120b
```

### Full Command with All Options

```bash
./run_pipeline.sh \
    --input_dir ./wildchat_data \
    --model_name gpt-oss-120b \
    --base_url http://localhost:8001/v1 \
    --api_key NA \
    --output_dir ./pipeline_output \
    --cuda_device 0
```

### Using Environment Variables

```bash
export BASE_URL="http://localhost:8001/v1"
export API_KEY="NA"

./run_pipeline.sh --input_dir ./wildchat_data
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input_dir` | Directory containing WildChat parquet files | Required |
| `--model_name` | Model name for generation/tracking | `gpt-oss-120b` |
| `--base_url` | Base URL for model API | `$BASE_URL` |
| `--api_key` | API key for model API | `$API_KEY` |
| `--output_dir` | Base output directory | `./pipeline_output` |
| `--cuda_device` | CUDA device ID for embedding model | `0` |
| `--help` | Show help message | - |

## Output Structure

After completion, you will find the following structure:

```
pipeline_output/
├── processed_data/              # Step 00: Preprocessed JSONL files
├── generated_prompts/           # Step 01: Preference prompts
├── <model_name>/                # Step 02: Model responses
├── <model_name>-filtered/       # Step 03: Filtered outputs
├── data/                        # Step 04: Merged conversations
├── hindsight/                   # Step 05: Hindsight mining results
├── refusal_prompts/             # Step 06: Refusal validation prompts
├── refusal_output/              # Step 06: Refusal validation responses
└── final/                       # Final output
    ├── train.jsonl              # Training set
    └── test.jsonl               # Test set (5000 samples)
```

## Individual Step Execution

If you need to run individual steps:

```bash
# Step 00: Preprocess
python3 00.preprocess_wildchat.py \
    --input_dir ./wildchat_data \
    --output_dir ./processed_data

# Step 01: Generate prompts
python3 01.gen_action.py \
    --input ./processed_data/file.jsonl \
    --output ./prompts/output.jsonl

# Step 02: Generate responses
python3 02.generate.py \
    --input_file ./prompts/input.jsonl \
    --save_dir ./responses \
    --save_name output.jsonl \
    --model_name_or_path gpt-oss-120b \
    --max_tokens 16384 \
    --temperature 0.5 \
    --num_threads 512 \
    --api_model \
    --n 1

# Step 03: Filter and parse
python3 03.gen_train.py \
    --input ./responses/input.jsonl \
    --output ./filtered/output.jsonl

# Step 04: Merge conversations
python3 04.merge.py \
    --input ./filtered/merged.jsonl \
    --output ./data/merge_convs.jsonl

# Step 05: Hindsight mining
python3 05.implicit_feedback_mining.py \
    --input ./data/merge_convs.jsonl \
    --output ./hindsight/added_data.jsonl \
    --cuda_device 0

# Step 06: Refusal validation (two sub-steps)
python3 06.gen_refusal_validation.py \
    --input ./filtered/merged.jsonl \
    --output ./refusal_prompts/prompts.jsonl

python3 02.generate.py \
    --input_file ./refusal_prompts/prompts.jsonl \
    --save_dir ./refusal_output \
    --save_name annotated.jsonl \
    --model_name_or_path gpt-oss-120b \
    --api_model \
    --n 1

# Step 07: Merge refusal validation
python3 07.merge_refusal_validation.py \
    --input ./filtered/merged.jsonl \
    --reann ./refusal_output/annotated.jsonl \
    --output ./final/final.jsonl

# Step 08: Train/test split
python3 08.split.py \
    --input1 ./final/final.jsonl \
    --input2 ./hindsight/added_data.jsonl \
    --train_output ./final/train.jsonl \
    --test_output ./final/test.jsonl \
    --test_count 5000
```

## Troubleshooting

### vLLM Server Issues

If vLLM fails to start:
- Check GPU memory: `nvidia-smi`
- Reduce `tensor-parallel-size` or use a smaller model
- Ensure you have enough disk space for model weights

### Out of Memory Errors

For Step 05 (embedding model):
- Reduce batch size or use CPU: set `--cuda_device -1`
- Use a smaller embedding model

### API Connection Errors

- Verify `BASE_URL` and `API_KEY` are correctly set
- Test API endpoint: `curl $BASE_URL/models`
- Check firewall settings

### Slow Processing

- Increase `--num_threads` for parallel API calls
- Use a faster GPU for vLLM
- Process data in smaller chunks

## Data Format

### Input Format (WildChat)

The pipeline expects WildChat parquet files with the following structure:
- `conversation`: List of conversation turns
- `language`: Language identifier (Chinese/English)
- `model`: Model name
- `conversation_hash`: Unique conversation identifier

### Output Format (Train/Test)

Final JSONL files contain:
```json
{
  "id": "unique_id",
  "history": [...],           # Conversation history
  "text": "prompt string",    # Evaluation prompt
  "messages": [...],          # Current turn messages
  "user_feedback": {...},     # User feedback
  "label": 1                  # Preference label (1-4)
}
```

**Label Mapping**:
- `1`: Clearly negative / rejection
- `2`: Correction / error pointer (negative)
- `3`: Positive engagement
- `4`: Clear satisfaction

## Contact

For questions or issues, please open a GitHub issue or contact [peng-h24@mails.tsinghua.edu.cn].
