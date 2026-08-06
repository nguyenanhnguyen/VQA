#!/usr/bin/env python3
"""
scripts/check_env.py
======================
Kiểm tra toàn bộ biến môi trường + kết nối cần thiết TRƯỚC khi service khởi
động thật. Chạy trong entrypoint.sh của Docker, hoặc chạy tay lúc setup:

    python scripts/check_env.py

"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def check_env() -> bool:
    print("Kiểm tra cấu hình môi trường VQA module...\n")
    ok = True

    # 1) Import settings -- việc này TỰ FAIL nếu thiếu biến bắt buộc (KEYFRAME_DIR...)
    #    nhờ field_validator trong config/settings.py, nên bọc try/except để in
    #    message rõ ràng thay vì traceback khó đọc.
    try:
        from config.settings import settings
        print("[OK] Đọc config/settings.py thành công (đủ biến bắt buộc).")
    except Exception as e:
        print(f"[FAIL] Thiếu/sai biến môi trường bắt buộc:\n  {e}")
        return False

    # 2) Kiểm tra các file dữ liệu có tồn tại thật trên đĩa không
    import os
    paths_to_check = {
        "KEYFRAME_DIR": settings.KEYFRAME_DIR,
        "CLIP_FEATURE_FILE": settings.CLIP_FEATURE_FILE,
        "SHOT_JSONL": settings.SHOT_JSONL,
        "WHISPER_JSONL": settings.WHISPER_JSONL,
        "TRANSCRIPT_JSONL": settings.TRANSCRIPT_JSONL,
    }
    for name, path in paths_to_check.items():
        if path and os.path.exists(path):
            print(f"[OK] {name} tồn tại: {path}")
        else:
            print(f"[WARN] {name} KHÔNG tìm thấy trên đĩa: {path} "
                  f"(có thể do volume chưa mount đúng trong docker run -v)")
            ok = False

    # 3) Kiểm tra kết nối Milvus (không raise cứng, chỉ cảnh báo -- vì có thể
    #    module đang test phần không liên quan Milvus)
    try:
        from pymilvus import connections
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT, timeout=5)
        print(f"[OK] Kết nối Milvus thành công: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        connections.disconnect("default")
    except Exception as e:
        print(f"[WARN] Không kết nối được Milvus tại {settings.MILVUS_HOST}:{settings.MILVUS_PORT}: {e}")
        ok = False

    # 4) In tóm tắt config đang dùng (che giấu key nhạy cảm)
    print("\n--- Tóm tắt cấu hình đang dùng ---")
    print(f"  EMBEDDING_MODEL_NAME = {settings.EMBEDDING_MODEL_NAME} (dim={settings.EMBEDDING_DIM})")
    print(f"  MILVUS = {settings.MILVUS_HOST}:{settings.MILVUS_PORT} / collection={settings.MILVUS_COLLECTION}")
    print(f"  LLM_MODEL_NAME = {settings.LLM_MODEL_NAME}")
    masked_key = ("*" * 6 + settings.LLM_API_KEY[-4:]) if settings.LLM_API_KEY and settings.LLM_API_KEY != "EMPTY" else settings.LLM_API_KEY
    print(f"  LLM_API_KEY = {masked_key}")
    print(f"  USE_GPU = {settings.USE_GPU} (device {settings.GPU_DEVICE})")

    return ok


if __name__ == "__main__":
    success = check_env()
    if success:
        print("\n✅ Tất cả kiểm tra bắt buộc đều PASS. Module sẵn sàng chạy.")
        sys.exit(0)
    else:
        print("\n❌ Có mục chưa đạt (xem [WARN]/[FAIL] ở trên). "
              "Sửa .env hoặc mount volume đúng rồi chạy lại script này trước khi start service.")
        sys.exit(1)
