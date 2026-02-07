import os
import itertools
from fastapi import FastAPI, Request
import httpx

# ==============================================================================
# Configuration (via Environment Variables)
# ==============================================================================

# Backend worker URLs - comma-separated list of worker endpoints
# Default: http://localhost:8004/get_reward,http://localhost:8005/get_reward,http://localhost:8006/get_reward,http://localhost:8007/get_reward
BACKENDS_STR = os.getenv(
    "BACKENDS",
    "http://localhost:8004/get_reward,http://localhost:8005/get_reward,http://localhost:8006/get_reward,http://localhost:8007/get_reward"
)
BACKENDS = [url.strip() for url in BACKENDS_STR.split(",") if url.strip()]

# Request timeout in seconds
# Default: 200
TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "200"))

# Router port
# Default: 9000
PORT = int(os.getenv("ROUTER_PORT", "9000"))

# ==============================================================================
# FastAPI App Setup
# ==============================================================================

app = FastAPI(title="Reward Router")

# Round-robin iterator
iterator = itertools.cycle(BACKENDS)

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def home():
    """Health check endpoint."""
    return {
        "message": "Reward Router is running",
        "backends": BACKENDS,
        "timeout": TIMEOUT
    }


@app.post("/get_reward")
async def router(request: Request):
    """
    Route incoming reward requests to backend workers using round-robin.
    """
    payload = await request.json()
    backend = next(iterator)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(backend, json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            return {
                "error": f"Backend request failed: {str(e)}",
                "backend": backend
            }
