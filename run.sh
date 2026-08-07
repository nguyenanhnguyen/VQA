#!/bin/bash
# =============================================================================
# run.sh - Helper script to build 1 time & run VQA module container
# =============================================================================

IMAGE_NAME="vqa-module:latest"
CONTAINER_NAME="vqa-module"
PORT=${PORT:-8000}
DATA_DIR=${DATA_ROOT:-"/data"}

# 1. Build image 1 lần nếu chưa tồn tại
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[run.sh] Image '$IMAGE_NAME' chưa tồn tại. Đang build image lần đầu..."
    docker build -t "$IMAGE_NAME" -f docker/Dockerfile .
else
    echo "[run.sh] Image '$IMAGE_NAME' đã tồn tại. Bỏ qua build."
fi

# 2. Xóa container cũ nếu đang chạy
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
    echo "[run.sh] Đang dừng và xóa container cũ..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# 3. Chạy container với mount code live & GPU support
echo "[run.sh] Khởi chạy container '$CONTAINER_NAME'..."
docker run --gpus all \
  -it --rm \
  --name "$CONTAINER_NAME" \
  -v "$(pwd)":/app \
  -v "$(pwd)/.cache":/app/.cache \
  -v "$DATA_DIR":/data \
  -p "$PORT":8000 \
  "$IMAGE_NAME" \
  uvicorn src.main:app --host 0.0.0.0 --port 8000 "$@"
