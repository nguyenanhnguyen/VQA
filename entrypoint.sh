#!/bin/bash
# =============================================================================
# entrypoint.sh - VQA Module Entrypoint
# =============================================================================
set -e

# Tự động cập nhật requirements.txt nếu có thay đổi mà KHÔNG CẦN rebuild Docker Image
if [ -f "/app/requirements.txt" ]; then
    mkdir -p /app/.cache
    HASH_FILE="/app/.cache/.requirements.hash"
    CURRENT_HASH=""
    if command -v md5sum >/dev/null 2>&1; then
        CURRENT_HASH=$(md5sum /app/requirements.txt | awk '{print $1}')
    elif command -v sha256sum >/dev/null 2>&1; then
        CURRENT_HASH=$(sha256sum /app/requirements.txt | awk '{print $1}')
    fi

    PREV_HASH=""
    [ -f "$HASH_FILE" ] && PREV_HASH=$(cat "$HASH_FILE")

    if [ -n "$CURRENT_HASH" ] && [ "$CURRENT_HASH" != "$PREV_HASH" ]; then
        echo "[entrypoint] Phát hiện requirements.txt thay đổi. Đang cài đặt dependencies..."
        pip install --no-cache-dir -r /app/requirements.txt
        echo "$CURRENT_HASH" > "$HASH_FILE"
        echo "[entrypoint] Cài đặt dependencies hoàn tất!"
    else
        echo "[entrypoint] requirements.txt không thay đổi. Bỏ qua pip install."
    fi
fi

echo "[entrypoint] Kiểm tra cấu hình môi trường..."
python scripts/check_env.py "$@"

echo "[entrypoint] Khởi động service..."
exec "$@"
