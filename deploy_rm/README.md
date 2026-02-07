# WildReward Model Deployment

A distributed serving system for reward models with load balancing. This deployment setup spins up multiple worker processes (each on its own GPU) behind a round-robin router for high-throughput inference.

## Architecture

```
                    ┌─────────────────┐
                    │   Router (9000) │
                    │  (Round-Robin)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌─────▼─────┐ ┌──────▼───────┐
    │ Worker (8004) │ │Worker(8005)│ │Worker (8006) │
    │    GPU 0      │ │  GPU 1     │ │   GPU 2      │
    └───────────────┘ └───────────┘ └──────────────┘
```

## Features

- **Distributed Workers**: Run multiple reward model instances across different GPUs
- **Load Balancing**: Round-robin router distributes requests evenly
- **Configurable**: Environment-based configuration for easy deployment
- **Mixed Precision**: Optional FP16 inference for faster throughput
- **Batch Processing**: Configurable batch size for efficient GPU utilization
- **Multiple Model Types**: Supports regression, ordinal, and classification models

## Requirements

- Python 3.8+
- CUDA-capable GPU(s)
- PyTorch with CUDA support

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Model Path

Set the `MODEL_PATH` environment variable to point to your trained model:

```bash
export MODEL_PATH="/path/to/your/model"
```

Or copy the example configuration and edit it:

```bash
cp .env.example .env
# Edit .env to set MODEL_PATH and other options
```

### 3. Deploy (One-Click)

```bash
chmod +x deploy.sh
./deploy.sh
```

This will start:
- **4 workers** on ports 8004-8007 using GPUs 0-3
- **1 router** on port 9000

### 4. Test the API

```bash
curl -X POST http://localhost:9000/get_reward \
  -H "Content-Type: application/json" \
  -d '{"query": ["What is the capital of France?", "Explain quantum computing"]}'
```

Response:
```json
{"rewards": [3.2, 4.1]}
```

## Configuration

All configuration is done via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model checkpoint | `./model` |
| `BATCH_SIZE` | Inference batch size | `16` |
| `USE_FP16` | Enable mixed precision | `true` |
| `MAX_LENGTH` | Max sequence length | `4096` |
| `CUDA_VISIBLE_DEVICES` | GPU device ID | (all GPUs) |
| `PORT` | Worker port | `8000` |
| `BACKENDS` | Comma-separated worker URLs | localhost:8004-8007 |
| `ROUTER_PORT` | Router port | `9000` |
| `ROUTER_TIMEOUT` | Request timeout (seconds) | `200` |
| `NUM_WORKERS` | Number of workers to spawn | `4` |
| `START_GPU_ID` | Starting GPU ID | `0` |
| `WORKER_START_PORT` | Starting port for workers | `8004` |

## Deployment Options

### Option 1: One-Click Script (Recommended)

```bash
# Basic usage
MODEL_PATH=/path/to/model ./deploy.sh

# Custom configuration
NUM_WORKERS=8 START_GPU_ID=0 WORKER_START_PORT=8000 ./deploy.sh
```

### Option 2: Manual Deployment

Start workers manually:

```bash
# Worker 1 on GPU 0, port 8004
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model PORT=8004 \
  uvicorn serve_rank:app --host 0.0.0.0 --port 8004 &

# Worker 2 on GPU 1, port 8005
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/path/to/model PORT=8005 \
  uvicorn serve_rank:app --host 0.0.0.0 --port 8005 &

# ... repeat for additional workers
```

Start the router:

```bash
BACKENDS=http://localhost:8004/get_reward,http://localhost:8005/get_reward \
  ROUTER_PORT=9000 \
  uvicorn router:app --host 0.0.0.0 --port 9000
```

## API Reference

### Health Check

```bash
curl http://localhost:9000/
```

```json
{
  "message": "Reward Router is running",
  "backends": ["http://localhost:8004/get_reward", ...],
  "timeout": 200
}
```

### Get Rewards

```bash
POST /get_reward
Content-Type: application/json

{
  "query": ["text 1", "text 2", "text 3"]
}
```

```json
{
  "rewards": [3.5, 4.2, 2.8]
}
```

## Advanced Usage

### Scaling to More GPUs

To deploy with 8 workers on GPUs 0-7:

```bash
NUM_WORKERS=8 START_GPU_ID=0 WORKER_START_PORT=8000 ./deploy.sh
```

### Using Specific GPUs

To use GPUs 4, 5, 6, 7:

```bash
NUM_WORKERS=4 START_GPU_ID=4 ./deploy.sh
```

### Custom Batch Size

For models with larger memory requirements:

```bash
BATCH_SIZE=8 ./deploy.sh
```

### Disable FP16

For models that don't support mixed precision:

```bash
USE_FP16=false ./deploy.sh
```

## Troubleshooting

### Workers fail to start

1. Check if `MODEL_PATH` is correctly set
2. Verify GPU availability with `nvidia-smi`
3. Check worker logs: `cat worker_8004.log`

### Out of memory errors

1. Reduce `BATCH_SIZE`: `BATCH_SIZE=8 ./deploy.sh`
2. Disable FP16: `USE_FP16=false ./deploy.sh`
3. Reduce `MAX_LENGTH`: `MAX_LENGTH=2048 ./deploy.sh`

### Router connection errors

1. Verify all workers are running: `curl http://localhost:8004/`
2. Check that `BACKENDS` environment variable matches worker ports
3. Increase `ROUTER_TIMEOUT` for large batches

## Stopping Services

Press `Ctrl+C` in the terminal running `deploy.sh`, or manually kill processes:

```bash
kill $(cat router.pid)
for pid_file in worker_*.pid; do kill $(cat $pid_file); done
```

## File Structure

```
deploy_rm/
├── serve_rank.py      # Worker service (loads model, handles inference)
├── router.py          # Load balancing router
├── deploy.sh          # One-click deployment script
├── requirements.txt   # Python dependencies
├── .env.example       # Configuration template
└── README.md          # This file
```

## License

This project is part of the WildReward framework.
