#!/bin/bash
# =============================================================================
# One-Click Deployment Script for Reward Model Service
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Default values (can be overridden via environment variables)
NUM_WORKERS=${NUM_WORKERS:-4}
START_GPU_ID=${START_GPU_ID:-0}
WORKER_START_PORT=${WORKER_START_PORT:-8004}
ROUTER_PORT=${ROUTER_PORT:-9000}
MODEL_PATH=${MODEL_PATH:-"./model"}

# Export MODEL_PATH so workers can use it
export MODEL_PATH

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

print_header "Reward Model Deployment"

# Check if MODEL_PATH exists
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path does not exist: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable to your model checkpoint directory."
    exit 1
fi

print_info "Configuration:"
echo "  - Model Path: $MODEL_PATH"
echo "  - Workers: $NUM_WORKERS"
echo "  - GPU Range: $START_GPU_ID to $((START_GPU_ID + NUM_WORKERS - 1))"
echo "  - Worker Ports: $WORKER_START_PORT to $((WORKER_START_PORT + NUM_WORKERS - 1))"
echo "  - Router Port: $ROUTER_PORT"
echo ""

# =============================================================================
# Cleanup Function
# =============================================================================

cleanup() {
    print_info "Stopping all services..."
    # Kill workers
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        port=$((WORKER_START_PORT + i))
        pid_file="worker_${port}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            kill $pid 2>/dev/null || true
            rm -f "$pid_file"
        fi
    done
    # Kill router
    if [ -f "router.pid" ]; then
        pid=$(cat "router.pid")
        kill $pid 2>/dev/null || true
        rm -f "router.pid"
    fi
    print_info "All services stopped."
}

# Trap signals for cleanup
trap cleanup EXIT INT TERM

# =============================================================================
# Start Worker Function
# =============================================================================

start_worker() {
    local gpu_id=$1
    local port=$2

    print_info "Starting worker on GPU $gpu_id, port $port..."

    CUDA_VISIBLE_DEVICES=$gpu_id \
    PORT=$port \
    MODEL_PATH=$MODEL_PATH \
    uvicorn serve_rank:app \
        --host 0.0.0.0 \
        --port $port \
        > "worker_${port}.log" 2>&1 &

    local pid=$!
    echo $pid > "worker_${port}.pid"

    # Wait for worker to be ready
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/" > /dev/null 2>&1; then
            print_info "Worker on port $port is ready!"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done

    print_error "Worker on port $port failed to start. Check worker_${port}.log for details."
    return 1
}

# =============================================================================
# Start Workers
# =============================================================================

print_header "Starting Workers"

# Build backend list for router
BACKENDS=""

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    gpu_id=$((START_GPU_ID + i))
    port=$((WORKER_START_PORT + i))

    if ! start_worker $gpu_id $port; then
        print_error "Failed to start worker $i. Exiting."
        exit 1
    fi

    # Add to backend list
    if [ -n "$BACKENDS" ]; then
        BACKENDS="${BACKENDS},"
    fi
    BACKENDS="${BACKENDS}http://localhost:${port}/get_reward"
done

print_info "All workers started successfully!"

# =============================================================================
# Start Router
# =============================================================================

print_header "Starting Router"

print_info "Starting router on port $ROUTER_PORT..."

BACKENDS="$BACKENDS" \
ROUTER_PORT=$ROUTER_PORT \
uvicorn router:app \
    --host 0.0.0.0 \
    --port $ROUTER_PORT \
    > "router.log" 2>&1 &

ROUTER_PID=$!
echo $ROUTER_PID > "router.pid"

# Wait for router to be ready
sleep 2
if curl -s "http://localhost:$ROUTER_PORT/" > /dev/null 2>&1; then
    print_info "Router is ready!"
else
    print_error "Router failed to start. Check router.log for details."
    exit 1
fi

# =============================================================================
# Deployment Complete
# =============================================================================

print_header "Deployment Complete!"

echo ""
print_info "Service URLs:"
echo "  - Router: http://localhost:$ROUTER_PORT"
echo "  - Health Check: http://localhost:$ROUTER_PORT/"
echo ""
print_info "API Usage:"
cat << 'EOF'
  curl -X POST http://localhost:9000/get_reward \
    -H "Content-Type: application/json" \
    -d '{"query": ["Your text here", "Another text"]}'
EOF
echo ""
print_info "Log Files:"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    port=$((WORKER_START_PORT + i))
    echo "  - Worker $port: worker_${port}.log"
done
echo "  - Router: router.log"
echo ""
print_info "To stop all services, press Ctrl+C or run:"
echo "  kill \$(cat router.pid)"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    port=$((WORKER_START_PORT + i))
    echo "  kill \$(cat worker_${port}.pid)"
done
echo ""

# Keep script running
print_info "Services are running. Press Ctrl+C to stop."
wait
