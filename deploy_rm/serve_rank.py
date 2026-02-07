import time
import os
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# 1. Configuration (via Environment Variables)
# ==============================================================================

# GPU device to use (e.g., "0", "1", "2", etc.)
# Default: unset (uses all available GPUs)
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")

# Model path - points to your trained model checkpoint
# Default: ./model (user should override this)
MODEL_PATH = os.getenv("MODEL_PATH", "./model")

# Inference device - auto-detect CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch size for inference - adjust based on GPU memory
# Default: 16
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# Enable FP16 (mixed precision) for faster inference and reduced memory usage
# Default: true
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

# Maximum sequence length for tokenization
# Default: 4096
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "4096"))

# Server port (optional, mainly for logging purposes)
PORT = int(os.getenv("PORT", "8000"))

# Set CUDA device if specified
if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# ==============================================================================
# 2. Load Model & Tokenizer
# ==============================================================================

print(f"Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

print(f"Loading model from: {MODEL_PATH}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()
print(f"Model loaded successfully on {DEVICE} (batch_size={BATCH_SIZE}, fp16={USE_FP16})")

# ==============================================================================
# 3. FastAPI App Setup
# ==============================================================================

app = FastAPI(
    title="Reward Model API",
    description="API for computing reward scores from (query, response) text pairs.",
    version="2.0"
)

# ==============================================================================
# 4. API Endpoints
# ==============================================================================

@app.get("/")
async def home():
    """Health check endpoint."""
    return {
        "message": "Reward Model API is running",
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "batch_size": BATCH_SIZE,
        "fp16": USE_FP16
    }


@app.post("/get_reward")
async def get_reward(request: Request):
    data = await request.json()
    texts = data.get("query")
    if texts is None:
        return {"error": "Missing 'query' field in request body"}

    results = []
    print(f"Received {len(texts)} texts for prediction.")

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)

            if USE_FP16 and DEVICE == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            logits = outputs.logits

            # Case 1: Regression model (output shape: batch_size, 1)
            if logits.shape[-1] == 1:
                preds = logits.squeeze(-1).cpu().float().tolist()

            # Case 2: CORAL / Ordinal Regression (output shape: batch_size, K-1)
            elif logits.shape[-1] in [3, 4]:  # Assuming 5 rating levels
                probs = torch.sigmoid(logits)
                expected_scores = 1 + torch.sum(probs, dim=-1)
                preds = expected_scores.cpu().float().tolist()

            # Case 3: Standard classification (output shape: batch_size, num_classes)
            else:
                preds = (torch.argmax(logits, dim=-1) + 1).cpu().tolist()

            results.extend(preds)

            del inputs, logits, outputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print(f"Finished prediction for {len(texts)} texts.")
    return {"rewards": results}
