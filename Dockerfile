# =============================================================================
# Dockerfile - VQA Module
#
# Đặt tên image theo quy định trường: <Nhóm>_<task>-<user>_<model>_<version>
# =============================================================================

# Base image khớp SẴN với torch==2.8.0 mà requirements.txt yêu cầu (Qwen3-VL-*
# chính thức pin torch==2.8.0) -- tránh việc pip install phải tải đè lại toàn
# bộ torch (mất thời gian, dễ lệch CUDA runtime nếu base image quá cũ).
# Đổi tag cuda nếu `nvidia-smi` trên server báo driver thấp hơn 12.4.
FROM pytorch/pytorch:2.8.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

RUN mkdir -p /data /app/logs

# ENTRYPOINT chạy check_env.py trước, "exec $@" mới chuyển giao cho CMD --
# khác với bản gốc (ENTRYPOINT + CMD tách rời khiến uvicorn KHÔNG BAO GIỜ
# được gọi, vì Docker chỉ nối CMD làm argv cho ENTRYPOINT, không chạy tuần tự).
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
