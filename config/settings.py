"""
config/settings.py
====================
Nguyên tắc: MỌI thông số kết nối/đường dẫn/tên model đều qua biến môi trường
(.env). Không hard-code trong code. Field bắt buộc sẽ làm module FAIL NGAY
LÚC IMPORT với message rõ ràng nếu thiếu -- đúng nguyên tắc "Zero-config
nhưng Easy-config" (fail-fast).
"""

from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True)

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Paths (dữ liệu đã index từ Layer 1/2/3) ---
    # Không có default = BẮT BUỘC phải có trong .env, module raise lỗi rõ ràng
    # ngay lúc khởi động nếu thiếu -- không âm thầm chạy sai.
    DATA_ROOT: str = "/data"
    KEYFRAME_DIR: Optional[str] = None
    CLIP_FEATURE_FILE: Optional[str] = None
    OBJECT_JSON_DIR: Optional[str] = None
    METADATA_DIR: Optional[str] = None
    SHOT_JSONL: Optional[str] = None
    WHISPER_JSONL: Optional[str] = None
    TRANSCRIPT_JSONL: Optional[str] = None

    # --- Milvus (Vector DB) --- default localhost cho dev, DLong đổi lại khi deploy
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "keyframe_embeddings"
    MILVUS_VECTOR_FIELD: str = "embedding"
    MILVUS_ID_FIELD: str = "id"
    MILVUS_METADATA_FIELDS: str = "video_id,frame_id,timestamp"

    # --- Embedding Model (BẮT BUỘC khớp đúng model đã dùng để tạo vector trong Milvus) ---
    # Default ViT-B-32: CLIP nhẹ, chạy được ngay trên máy dev bất kỳ, không cần
    # checkpoint đặc biệt -- mục tiêu "zero-config chạy thử được ngay".
    # Khi DLong xác nhận Milvus dùng model khác (vd Qwen3-VL-Embedding-8B), chỉ
    # cần đổi 2 dòng dưới trong .env, KHÔNG cần sửa code.
    EMBEDDING_MODEL_NAME: str = "ViT-B-32"
    EMBEDDING_DIM: int = 512

    # --- LLM / VLM ---
    LLM_API_BASE: Optional[str] = None   # vd http://localhost:8000/v1 nếu dùng vLLM/Ollama nội bộ
    LLM_API_KEY: Optional[str] = None    # "EMPTY" nếu server nội bộ không cần key
    LLM_MODEL_NAME: str = "Qwen2.5-7B-Instruct"
    VLM_MODEL_NAME: str = "Qwen/Qwen2-VL-7B-Instruct"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Thresholds (placeholder, hiệu chỉnh lại bằng benchmark thật VQA-T07) ---
    CONFIDENCE_MARGIN_THRESHOLD: float = 0.15
    CONFIDENCE_AGREEMENT_THRESHOLD: float = 0.6

    # --- Tier Rules ---
    TIER0_MAX_ENTITIES: int = 2
    TIER0_MAX_RELATIONS: int = 1

    # --- Performance ---
    MAX_BATCH_SIZE: int = 32
    USE_GPU: bool = True
    GPU_DEVICE: int = 0

    # --- Versioning (VQA-T09/T13) ---
    MODEL_VERSION_TAG: str = "vqa-baseline-dev"

    # ------------------------------------------------------------------
    # Validators: field name khớp thẳng với tên biến môi trường cùng tên
    # (pydantic-settings tự map theo tên, KHÔNG cần Field(..., env="X")
    # như pydantic v1 -- cú pháp đó không hợp lệ với pydantic-settings v2).
    # ------------------------------------------------------------------
    @field_validator(
        "KEYFRAME_DIR", "CLIP_FEATURE_FILE", "OBJECT_JSON_DIR", "METADATA_DIR",
        "SHOT_JSONL", "WHISPER_JSONL", "TRANSCRIPT_JSONL",
    )
    @classmethod
    def require_path(cls, v, info):
        if not v:
            raise ValueError(
                f"Thiếu biến môi trường bắt buộc: {info.field_name}. "
                f"Điền vào file .env (xem .env.example) trước khi chạy."
            )
        return v

    @field_validator("MILVUS_HOST", "MILVUS_COLLECTION")
    @classmethod
    def not_empty(cls, v, info):
        if not v:
            raise ValueError(f"{info.field_name} không được để trống. Kiểm tra .env")
        return v


# Singleton -- import 1 lần, dùng lại khắp nơi. Validate (fail-fast) xảy ra
# NGAY DÒNG NÀY, nên chỉ cần `from config.settings import settings` là biết
# ngay .env có thiếu gì hay không, không cần đợi tới lúc gọi API mới lỗi.
settings = Settings()
