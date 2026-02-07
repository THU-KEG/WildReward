# Online DPO with verl using WildReward

This project extends [VERL](https://github.com/verl-project/verl) (Volcano Engine Reinforcement Learning for LLMs) to implement **Online DPO** (Direct Preference Optimization) training for language models. It enables training with remote reward models through a flexible API-based architecture.

## Overview

- **Base Framework**: Built on top of [verl-project/verl](https://github.com/verl-project/verl)
- **Key Feature**: Online DPO training module with remote reward model integration
- **Reward Backend**: Configurable remote API for reward scoring (see `verl/utils/reward_score/remote_wild_rm.py`)

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU(s)
- 8+ GPUs recommended for training (configurable)

---

### Step 1: Environment Setup

Set up your environment following the official [VERL documentation](https://github.com/verl-project/verl).

```bash
# Clone and install dependencies
pip install -e .
```

Required dependencies include:
- PyTorch with CUDA support
- Ray (for distributed training)
- Transformers
- vLLM (for rollout generation)
- Additional dependencies in `requirements.txt`

---

### Step 2: Deploy Your Reward Model

Deploy your reward model as a web service and obtain its API endpoint.

**Configure the API URL:**

Edit `verl/utils/reward_score/remote_wild_rm.py` and set your reward model URL:

```python
# Line 10 in remote_wild_rm.py
api_url = "https://your-reward-model-endpoint.com/api"  # Replace with your URL
```

**Reward Model API Requirements:**

Your reward model should accept POST requests with the following format:

```json
{
  "query": ["<prompt_text_1>", "<prompt_text_2>", ...]
}
```

And return responses in this format:

```json
{
  "rewards": [3.5, 4.2, ...]  // List of reward scores
}
```

---

### Step 3: Prepare Training Prompts

Prepare your training data as a JSONL file where each line contains a prompt:

```jsonl
{"prompt": "What is the capital of France?"}
{"prompt": "Explain quantum computing in simple terms."}
```

**Process the prompts into parquet format:**

```bash
cd examples/data_preprocess
python prompts.py --data_path <path_to_your_jsonl_file> --local_dir ~/data/general_domain
```

This will:
1. Load your prompts from the JSONL file
2. Transform them into the required format
3. Save the dataset as `train.parquet` in the specified directory

---

### Step 4: Run Training

Once your environment is configured, reward model is deployed, and training data is ready:

```bash
cd examples/online_dpo_trainer
bash run.sh
```

**Or customize your training directly:**

```bash
bash run_llama3_8b.sh
```

Edit `run_llama3_8b.sh` to adjust:
- Model paths (`actor_rollout_ref.model.path`)
- Data paths (`data.train_files`, `data.val_files`)
- Batch sizes and hyperparameters
- GPU configuration

---

## Configuration

### Training Hyperparameters

Key parameters in `run_llama3_8b.sh`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `algorithm.adv_estimator` | Algorithm type | `online_dpo` |
| `algorithm.kl_ctrl.kl_coef` | KL divergence coefficient | `0.1` |
| `data.train_batch_size` | Training batch size | `64` |
| `data.max_prompt_length` | Max prompt tokens | `1024` |
| `data.max_response_length` | Max response tokens | `4096` |
| `actor_rollout_ref.rollout.n` | Number of rollouts per prompt | `8` |

### Reward Scoring

The reward scoring is handled in `verl/utils/reward_score/remote_wild_rm.py`:

- **Batch processing**: Configurable batch size and thread count
- **Error handling**: Default score (2.0) on API failures
- **Prompt template**: Built-in chat evaluation template

---

## Project Structure

```
.
├── verl/
│   └── utils/
│       └── reward_score/
│           └── remote_wild_rm.py    # Remote reward model integration
├── examples/
│   ├── data_preprocess/
│   │   └── prompts.py                # Data preprocessing script
│   └── online_dpo_trainer/
│       ├── run.sh                    # Quick start script
│       └── run_llama3_8b.sh          # Example training config
└── README.md
```

---

## Troubleshooting

**Issue**: `ValueError: You need set the api_url for your remote reward model first.`

**Solution**: Make sure you've set `api_url` in `verl/utils/reward_score/remote_wild_rm.py`

**Issue**: CUDA out of memory

**Solution**: Reduce `data.max_response_length` or `actor_rollout_ref.rollout.n` in the training script

---

## License

This project inherits the license from the [VERL project](https://github.com/verl-project/verl). See [LICENSE](LICENSE) for details.
