"""
config/settings.py
====================
Cấu hình hệ thống VQA Module.
Hỗ trợ cả .env, smart default từ DATA_ROOT, và ghi đè bằng CLI arguments:
  --data / --data-root
  --keyframe / --keyframe-dir
  --ocr / --ocr-dir
  --object / --object-json-dir
  --metadata / --metadata-dir
  --shot / --shot-jsonl
  --whisper / --whisper-jsonl
  --transcript / --transcript-jsonl
  --clip-feature / --clip-feature-file
"""

import os
import sys
import argparse
from typing import Optional, List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Paths (dữ liệu đã index từ Layer 1/2/3) ---
    DATA_ROOT: str = "/data"
    KEYFRAME_DIR: Optional[str] = None
    CLIP_FEATURE_FILE: Optional[str] = None
    OBJECT_JSON_DIR: Optional[str] = None
    METADATA_DIR: Optional[str] = None
    SHOT_JSONL: Optional[str] = None
    WHISPER_JSONL: Optional[str] = None
    TRANSCRIPT_JSONL: Optional[str] = None
    OCR_DIR: Optional[str] = None

    # --- Milvus (Vector DB) ---
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "keyframe_embeddings"
    MILVUS_VECTOR_FIELD: str = "embedding"
    MILVUS_ID_FIELD: str = "id"
    MILVUS_METADATA_FIELDS: str = "video_id,frame_id,timestamp"

    # --- Embedding Model ---
    EMBEDDING_MODEL_NAME: str = "ViT-B-32"
    EMBEDDING_DIM: int = 512

    # --- LLM / VLM ---
    LLM_API_BASE: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: str = "Qwen2.5-7B-Instruct"
    VLM_MODEL_NAME: str = "Qwen/Qwen2-VL-7B-Instruct"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Cascade & Query Refinement Settings ---
    ENABLE_QUERY_REFINEMENT: bool = True
    GATE_B_CONFIDENCE_THRESHOLD: float = 0.65
    ENABLE_LIGHT_VERIFY: bool = True

    # --- Thresholds ---
    CONFIDENCE_MARGIN_THRESHOLD: float = 0.15
    CONFIDENCE_AGREEMENT_THRESHOLD: float = 0.6

    # --- Tier Rules ---
    TIER0_MAX_ENTITIES: int = 2
    TIER0_MAX_RELATIONS: int = 1

    # --- Performance ---
    MAX_BATCH_SIZE: int = 32
    USE_GPU: bool = True
    GPU_DEVICE: int = 0

    # --- Versioning ---
    MODEL_VERSION_TAG: str = "vqa-baseline-dev"

    def auto_fill_paths(self):
        """Tự động điền các đường dẫn con dựa vào DATA_ROOT nếu chưa được chỉ định"""
        root = self.DATA_ROOT
        if not self.KEYFRAME_DIR:
            self.KEYFRAME_DIR = os.path.join(root, "keyframes")
        if not self.CLIP_FEATURE_FILE:
            self.CLIP_FEATURE_FILE = os.path.join(root, "clip_features.npy")
        if not self.OBJECT_JSON_DIR:
            self.OBJECT_JSON_DIR = os.path.join(root, "objects")
        if not self.METADATA_DIR:
            self.METADATA_DIR = os.path.join(root, "metadata")
        if not self.SHOT_JSONL:
            self.SHOT_JSONL = os.path.join(root, "shots.jsonl")
        if not self.WHISPER_JSONL:
            self.WHISPER_JSONL = os.path.join(root, "whisper.jsonl")
        if not self.TRANSCRIPT_JSONL:
            self.TRANSCRIPT_JSONL = os.path.join(root, "shot_transcripts.jsonl")
        if not self.OCR_DIR:
            self.OCR_DIR = os.path.join(root, "ocr")

    @field_validator("MILVUS_HOST", "MILVUS_COLLECTION")
    @classmethod
    def not_empty(cls, v, info):
        if not v:
            raise ValueError(f"{info.field_name} không được để trống. Kiểm tra .env")
        return v


# Khởi tạo singleton settings
settings = Settings()
settings.auto_fill_paths()


def apply_cli_overrides(args_list: Optional[List[str]] = None) -> Settings:
    """
    Parse command-line flags và ghi đè trực tiếp vào object `settings`.
    Cho phép gọi: python -m src.main --data /my/data --keyframe /my/keys --ocr /my/ocr
    """
    if args_list is None:
        args_list = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", "--data-root", dest="data_root", type=str, default=None)
    parser.add_argument("--keyframe", "--keyframe-dir", dest="keyframe_dir", type=str, default=None)
    parser.add_argument("--ocr", "--ocr-dir", dest="ocr_dir", type=str, default=None)
    parser.add_argument("--object", "--object-json-dir", dest="object_json_dir", type=str, default=None)
    parser.add_argument("--metadata", "--metadata-dir", dest="metadata_dir", type=str, default=None)
    parser.add_argument("--shot", "--shot-jsonl", dest="shot_jsonl", type=str, default=None)
    parser.add_argument("--whisper", "--whisper-jsonl", dest="whisper_jsonl", type=str, default=None)
    parser.add_argument("--transcript", "--transcript-jsonl", dest="transcript_jsonl", type=str, default=None)
    parser.add_argument("--clip-feature", "--clip-feature-file", dest="clip_feature_file", type=str, default=None)

    parsed, _ = parser.parse_known_args(args_list)

    if parsed.data_root:
        settings.DATA_ROOT = parsed.data_root
        # Re-fill default subpaths based on new DATA_ROOT
        settings.auto_fill_paths()

    if parsed.keyframe_dir:
        settings.KEYFRAME_DIR = parsed.keyframe_dir
    if parsed.ocr_dir:
        settings.OCR_DIR = parsed.ocr_dir
    if parsed.object_json_dir:
        settings.OBJECT_JSON_DIR = parsed.object_json_dir
    if parsed.metadata_dir:
        settings.METADATA_DIR = parsed.metadata_dir
    if parsed.shot_jsonl:
        settings.SHOT_JSONL = parsed.shot_jsonl
    if parsed.whisper_jsonl:
        settings.WHISPER_JSONL = parsed.whisper_jsonl
    if parsed.transcript_jsonl:
        settings.TRANSCRIPT_JSONL = parsed.transcript_jsonl
    if parsed.clip_feature_file:
        settings.CLIP_FEATURE_FILE = parsed.clip_feature_file

    return settings
