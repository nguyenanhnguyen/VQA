#!/bin/bash
# entrypoint.sh
# ==============
# Chạy check_env.py trước để fail-fast rõ ràng nếu thiếu config, CHỈ SAU KHI
# check thành công mới "exec" sang lệnh thật (uvicorn) truyền vào CMD.
#
# "exec" (không phải chạy như subprocess con) quan trọng vì nó thay thế hẳn
# tiến trình shell hiện tại bằng uvicorn -- giữ đúng PID 1, để Docker gửi được
# tín hiệu dừng (SIGTERM khi `docker stop`) thẳng tới uvicorn thay vì bị kẹt
# ở tiến trình bash trung gian.

set -e

echo "[entrypoint] Kiểm tra cấu hình môi trường..."
python scripts/check_env.py

echo "[entrypoint] Cấu hình hợp lệ. Khởi động service..."
exec "$@"
