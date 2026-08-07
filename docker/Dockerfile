# =============================================================================
# Dockerfile - VQA Module (Optimized for RTX 4050 6GB VRAM & CUDA 12.4)
# =============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME="/app/.cache/huggingface" \
    TORCH_HOME="/app/.cache/torch"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and pre-install dependencies into image layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create needed directories
RUN mkdir -p /data /app/logs /app/.cache

RUN chmod +x entrypoint.sh run.sh 2>/dev/null || true

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
