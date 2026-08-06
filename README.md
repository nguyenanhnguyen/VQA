```markdown
### Kiến trúc tổng quan

```
Query → Router Heuristic → Phân loại Tier
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
           Tier 0 (Đơn giản)       Tier 2 (Phức tạp)
                │                       │
                ▼                       ▼
        CLIP + RRF đơn giản     Agentic Pipeline đầy đủ
                │                       │
                └───────────┬───────────┘
                            ▼
                   Confidence & Decision
                            │
                    ┌───────┴───────┐
                    ▼               ▼
               Tự động nộp    Đề xuất review
               (Vòng loại)    (Vòng chung kết)
```

### Các Agent trong Tier 2

| Agent | Nhiệm vụ | Dữ liệu truy xuất | Trạng thái |
|-------|----------|-------------------|------------|
| **Vision Agent** | Tìm keyframe theo CLIP embedding | Milvus (vector) | ✅ Hoạt động |
| **ASR Agent** | Tìm shot có transcript khớp | `shot_transcripts.jsonl` | ✅ Hoạt động |
| **Metadata Agent** | Tìm video có metadata khớp | `metadata/*.json` | ✅ Hoạt động |
| **OCR Agent** | Tìm keyframe có văn bản khớp | OCR files | ⚠️ Placeholder |
| **Object Agent** | Tìm keyframe có object category | YOLO-World JSON | ⚠️ Placeholder |

---

## 📂 Cấu trúc thư mục

```
vqa_module/
├── .env.example          # Mẫu file cấu hình
├── .env                  # File cấu hình thực tế (tự tạo)
├── Dockerfile            # Build image
├── entrypoint.sh         # Entry point
├── requirements.txt      # Python dependencies
├── config/
│   ├── settings.py       # Cấu hình từ biến môi trường
│   └── prompts.yaml      # Prompt templates
├── scripts/
│   └── check_env.py      # Kiểm tra môi trường
├── src/                  # Source code
└── tests/                # Dữ liệu test (tùy chọn)
```

---

## ⚙️ Cấu hình (quan trọng)

### Bước 1: Tạo file `.env`

```bash
cp .env.example .env
```

### Bước 2: Sửa các biến môi trường

Mở `.env` và điền đúng thông tin của hệ thống bạn:

| Biến | Ý nghĩa | Ví dụ (thay bằng giá trị thực tế) |
|------|---------|-----------------------------------|
| `DATA_ROOT` | Thư mục gốc chứa dữ liệu | `/data` |
| `KEYFRAME_DIR` | Thư mục chứa keyframe | `/data/keyframes` |
| `CLIP_FEATURE_FILE` | File CLIP embedding `.npy` | `/data/clip_features.npy` |
| `OBJECT_JSON_DIR` | Thư mục chứa object detection JSON | `/data/objects` |
| `METADATA_DIR` | Thư mục chứa metadata JSON | `/data/metadata` |
| `SHOT_JSONL` | File shots.jsonl (từ Indexing Pipeline) | `/data/shots.jsonl` |
| `WHISPER_JSONL` | File whisper.jsonl (từ Indexing Pipeline) | `/data/whisper.jsonl` |
| `TRANSCRIPT_JSONL` | File shot_transcripts.jsonl | `/data/shot_transcripts.jsonl` |
| `MILVUS_HOST` | Địa chỉ Milvus server | `localhost` hoặc `milvus-standalone` |
| `MILVUS_PORT` | Cổng Milvus | `19530` |
| `MILVUS_COLLECTION` | Tên collection trong Milvus | `keyframe_embeddings` |
| `EMBEDDING_MODEL_NAME` | **Model đã dùng để tạo vector trong Milvus** | `ViT-B-32` |
| `EMBEDDING_DIM` | Số chiều của vector (phải khớp với dữ liệu) | `512` |
| `LLM_API_BASE` | Endpoint LLM nội bộ (nếu dùng local) | `http://localhost:8000/v1` |
| `LLM_API_KEY` | API key (nếu dùng OpenAI) | `sk-...` hoặc `EMPTY` |
| `LOG_LEVEL` | Mức độ log | `INFO` hoặc `DEBUG` |

**⚠️ Quan trọng:** `EMBEDDING_MODEL_NAME` và `EMBEDDING_DIM` phải khớp chính xác với model đã dùng để tạo vector và insert vào Milvus. Nếu sai, kết quả truy xuất sẽ sai hoàn toàn.

---

## 🐳 Chạy với Docker

### Build image

```bash
docker build -t vqa-module:latest .
```

### Chạy container

```bash
docker run --gpus all \
  -v /path/to/real/data:/data \
  -v $(pwd)/.env:/app/.env \
  -p 8021:8000 \
  vqa-module:latest \
  uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Giải thích:**
- `-v /path/to/real/data:/data`: mount thư mục dữ liệu vào container.
- `-v $(pwd)/.env:/app/.env`: truyền file cấu hình.
- `-p 8021:8000`: mở cổng 8021 bên ngoài (có thể đổi).

### Kiểm tra container đã chạy

```bash
docker ps
```

Nếu container có trạng thái `Up`, module đã chạy thành công.

---

## 🧪 Test API

### 1. Health check

```bash
curl http://localhost:8021/health
```

Kết quả kỳ vọng: `{"status": "ok"}`

### 2. Gửi câu hỏi VQA

```bash
curl -X POST http://localhost:8021/vqa/query \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "question": "Người đàn ông mặc áo đỏ đang làm gì?",
    "max_results": 10
  }'
```

### 3. Xem docs (Swagger UI)

Mở trình duyệt truy cập:
```
http://localhost:8021/docs
```

Tại đây bạn có thể xem tất cả endpoint và test trực tiếp.

---

## 🔍 Checklist trước khi test

| Hạng mục | Kiểm tra |
|----------|----------|
| Milvus đang chạy và có collection với dữ liệu | `docker ps` |
| File `.env` đã điền đúng đường dẫn | `cat .env` |
| Các file JSONL đã tồn tại | `ls -la /data/*.jsonl` |
| Keyframe và CLIP features đã có | `ls -la /data/keyframes/` |
| Container VQA đang chạy | `docker ps` |

Nếu thiếu bất kỳ mục nào, module sẽ báo lỗi khi gọi API.

---

## 🛠️ Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|-----|-------------|----------------|
| `ModuleNotFoundError: No module named 'xxx'` | Thiếu thư viện Python | Cài lại: `pip install -r requirements.txt` |
| `Failed to connect to Milvus` | Milvus chưa chạy hoặc sai host/port | Kiểm tra Milvus: `docker ps`, sửa `.env` |
| `Collection not found` | Sai tên collection | Kiểm tra collection trong Milvus: `docker exec -it milvus bash` rồi dùng `pymilvus` |
| `File not found: /data/xxx` | Đường dẫn sai hoặc chưa mount đúng | Kiểm tra `-v` khi chạy container, sửa `.env` |
| `CUDA out of memory` | Model quá lớn so với VRAM | Dùng model nhỏ hơn hoặc giảm batch size |
| `API trả về candidates rỗng` | Chưa có dữ liệu hoặc embedding model sai | Kiểm tra `EMBEDDING_MODEL_NAME` và dữ liệu trong Milvus |

---

## 📊 Trạng thái hiện tại của module

| Thành phần | Trạng thái | Ghi chú |
|------------|------------|---------|
| Router (Tier 0/2) | ✅ Hoạt động | Heuristic rule-based |
| Tier 0 (CLIP + RRF) | ✅ Hoạt động | Đã kiểm tra |
| Vision Agent | ✅ Hoạt động | Dùng CLIP trong Milvus |
| ASR Agent | ✅ Hoạt động | Đọc transcript.jsonl |
| Metadata Agent | ✅ Hoạt động | Đọc metadata JSON |
| Fusion (RRF) | ✅ Hoạt động | Đã kiểm tra |
| Reranker (Cross-Encoder) | ✅ Hoạt động | Đã kiểm tra |
| Answer Generation (VLM) | ✅ Hoạt động | Cần Qwen2-VL-7B |
| Planner + Reflection | ✅ Hoạt động | Cần LLM endpoint |
| OCR Agent | ⚠️ Placeholder | Cần dữ liệu OCR thực |
| Object Agent | ⚠️ Placeholder | Cần dữ liệu Object thực |

**OCR Agent và Object Agent đang ở dạng placeholder**, do đó:
- Các câu hỏi yêu cầu **đọc chữ trong video** (biển báo, tiêu đề, số đếm) sẽ không có kết quả tốt.
- Các câu hỏi yêu cầu **nhận diện vật thể cụ thể** (người, xe, đồ vật) cũng sẽ hạn chế.

Nếu cần sử dụng đầy đủ chức năng, hãy hoàn thiện 2 agent này với dữ liệu thực.
