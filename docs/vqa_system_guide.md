# Hướng dẫn Kiến trúc Cascade & Tùy chỉnh VQA Module

Tài liệu này giải thích chi tiết kiến trúc **Cascade Redesign (Phiên bản 2.0)** của VQA Module, tích hợp **Bilingual Query Refinement** (Song ngữ Việt - Anh), cơ chế phân luồng hai cổng **Gate A / Gate B**, tối ưu cho GPU RTX 4050 6GB VRAM, cơ chế 1-time Docker build, các tham số dòng lệnh `--data` / `--keyframe` / `--ocr`, và hướng dẫn tùy chỉnh thủ công.

---

## 1. Kiến trúc Cascade Redesign (System Architecture)

Luồng xử lý một truy vấn VQA (`POST /vqa/query`):

```
Query (Tiếng Việt hoặc Tiếng Anh)
      │
      ▼
Bilingual Query Refiner (src/query_refiner.py)
      ├── visual_query_en: Prompt rút gọn tiếng Anh ──► CLIP Embedder (Milvus)
      ├── visual_query_vi: Prompt rút gọn tiếng Việt ──► Multilingual Embedder
      └── keywords_vi / keywords_en ──────────────────► ASR / OCR / Metadata Agents
      │
      ▼
Gate A Router (src/router.py) ── Phân loại intent câu hỏi
      │
      ├── Structured Evidence ("tier0")
      │     │
      │     ▼
      │   Tier 0 Search: Multi-Agent Retrieval
      │           + Fast Text-LLM Fact Synthesizer (KHÔNG gọi VLM -> Độ trễ <1s)
      │     │
      │     ▼
      │   Gate B Confidence Evaluator (src/confidence.py)
      │     │
      │     ├── Đủ độ tin cậy (High Confidence) ──────────► Response (Auto-Submit)
      │     │
      │     └── Thiếu độ tin cậy (Low Confidence)
      │           │
      │           ▼
      │     Tier 2 Escalation
      │           │
      │           ▼
      │     Light Verify (src/tier2/light_verify.py): Kiểm tra VLM 1 khung hình nhanh
      │           │
      │           ├── Xác minh thành công (Verified) ────► Response
      │           │
      │           └── Không xác minh được (Unsure) ───► Full Agentic Pipeline (Tier 2)
      │                                                     │
      │                                                     ▼
      └── Visual Reasoning ("tier2") ───────────────────────► Response
```

### Các thành phần chính:

1. **Bilingual Query Refiner (`src/query_refiner.py`)**:
   - Tự động phát hiện ngôn ngữ (Tiếng Việt / Tiếng Anh).
   - Rút gọn câu hỏi giao tiếp thành prompt trực quan ngắn gọn (`visual_query_en` & `visual_query_vi`), tăng độ tương đồng cosine trong không gian vector CLIP.
   - Trích xuất từ khóa tìm kiếm (`keywords_vi` & `keywords_en`) cho các agent ASR, OCR, Metadata, và Object.

2. **Gate A Router (`src/router.py`)**:
   - Phân loại câu hỏi thành `tier0` (câu hỏi trả lời được từ dữ liệu cấu trúc: chữ OCR, lời thoại ASR, metadata, số lượng vật thể) hoặc `tier2` (câu hỏi cần suy luận thị giác/không gian phức tạp).

3. **Tier 0 Fast Fact Synthesizer (`src/tier0.py`)**:
   - Thực hiện tìm kiếm đa nguồn qua các Agent chuyên trách.
   - Dùng Text-only LLM tổng hợp câu trả lời tự nhiên từ bằng chứng thu được.
   - **Hoàn toàn KHÔNG gọi VLM mã hóa ảnh** -> Giảm latency xuống dưới 1 giây.

4. **Gate B Evidence Confidence Evaluator (`src/confidence.py`)**:
   - Đánh giá độ tin cậy kết quả Tier 0 dựa trên:
     $$\text{Confidence} = 0.4 \times \text{MarginScore} + 0.4 \times \text{AgreementScore} + 0.2 \times \text{CoverageScore}$$
   - Nếu `confidence < 0.65`, kích hoạt leo thang (escalation) sang Tier 2.

5. **Tier 2 Light Verify & Full Agentic Pipeline (`src/tier2/light_verify.py` & `src/tier2/iterative.py`)**:
   - **Light Verify**: Gọi VLM kiểm tra 1 khung hình duy nhất để xác minh câu trả lời đề xuất.
   - **Full Agentic Pipeline**: Chạy toàn bộ Planner + 5 Agents + Fusion + Reranker + VLM + Reflection nếu Light Verify không xác minh được.

---

## 2. Tối ưu cho RTX 4050 6GB VRAM

VRAM 6GB trên RTX 4050 Laptop cần quản lý bộ nhớ cẩn thận để tránh lỗi `CUDA out of memory`:

1. **Cấu hình Tránh Phân mảnh VRAM**:
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` được bật sẵn trong Dockerfile và runtime environment.
2. **Khuyến nghị Chạy Model VLM/LLM**:
   - Chạy LLM/VLM lớn (như Qwen2.5-7B hoặc Qwen2-VL-7B) qua **Ollama** hoặc **vLLM** trên host/server phụ (`LLM_API_BASE=http://localhost:11434/v1`), tránh load trực tiếp 14GB weights vào 6GB VRAM.
   - Nếu chạy local trên GPU 6GB, dùng model embedding nhẹ (`ViT-B-32`, ~350MB VRAM).

---

## 3. Cơ chế Build 1 Lần & Cập nhật `requirements.txt` Động

Bạn không cần rebuild image khi sửa `requirements.txt`:

1. **Cách hoạt động**:
   - File `entrypoint.sh` lưu mã MD5 hash của `requirements.txt` vào `/app/.cache/.requirements.hash`.
   - Mỗi lần chạy `./run.sh`, container kiểm tra xem `requirements.txt` có thay đổi hay không.
   - Nếu có thay đổi, `entrypoint.sh` tự chạy `pip install` cập nhật thư viện ngay trong container.
   - Nếu không thay đổi, bọc qua ngay để bắt đầu service siêu nhanh.

2. **Lệnh chạy**:
   ```bash
   ./run.sh
   ```

---

## 4. Sử dụng Tham số Dòng Lệnh `--data`, `--keyframe`, `--ocr`...

Bạn có thể truyền trực tiếp đường dẫn dữ liệu từ terminal:

### Đổi thư mục gốc (`--data` hoặc `--data-root`):
```bash
./run.sh --data /path/to/my_data
```
*(Các đường dẫn con `keyframes/`, `ocr/`, `objects/`, `metadata/`, `shots.jsonl` sẽ tự động chuyển sang `/path/to/my_data/...`)*

### Đổi từng đường dẫn riêng biệt:
```bash
./run.sh --keyframe /path/to/keyframes --ocr /path/to/ocr_folder --metadata /path/to/meta
```

---

## 5. Hướng dẫn Tùy chỉnh & Nâng cấp Thủ công

### A. Tùy chỉnh Ngưỡng Escalation của Gate B
Trong `config/settings.py` hoặc file `.env`:
```env
GATE_B_CONFIDENCE_THRESHOLD=0.65
```
Giảm xuống `0.5` nếu muốn chấp nhận nhiều câu trả lời Tier 0 hơn (nhanh hơn), tăng lên `0.75` nếu muốn cẩn trọng leo thang sang Tier 2 nhiều hơn.

### B. Thay đổi Endpoint LLM / VLM (Ví dụ: Chuyển sang Ollama)
Chạy Ollama trên máy với model `qwen2.5:7b`, sau đó chỉnh `.env`:
```env
LLM_API_BASE=http://host.docker.internal:11434/v1
LLM_API_KEY=EMPTY
LLM_MODEL_NAME=qwen2.5:7b
```

---
