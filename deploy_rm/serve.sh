
#!/bin/bash
# 启动 4 个 RM worker
for i in 4 5 6 7
do
  port=$((8000 + i))
  echo "Starting RM worker on port $port..."
  CUDA_VISIBLE_DEVICES=$i uvicorn serve_rank:app --host 0.0.0.0 --port $port &
done

# 启动 Router (9000)
echo "Starting router on port 9000..."
uvicorn router:app --host 0.0.0.0 --port 9000
