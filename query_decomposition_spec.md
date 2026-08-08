# Query Decomposition & Refinement Specification

## 1. Mục tiêu

Thay thế `QueryRefiner` hiện tại bằng một module **LLM-based Query Understanding + Query Decomposition** phục vụ trực tiếp cho pipeline Retrieval/VQA.

Mục tiêu chính:

1. Chuẩn hóa mọi câu hỏi về **English semantic representation**.
2. Phân tích câu hỏi để trích xuất các thành phần có ích cho các retriever/filter chuyên biệt:
   - Visual retrieval
   - OCR
   - ASR
   - Object Detection (OD)
   - Metadata
3. Với câu hỏi phức tạp, sử dụng LLM để **decompose** thành các sub-query đơn giản.
4. Phân biệt rõ hai loại output:
   - **Question-derived signals**: keyword/intent/entity/attribute dùng để xác định *cần tìm gì* và chọn modality/filter.
   - **Context/Hint-derived sub-queries**: các truy vấn đơn giản dùng trực tiếp cho retrieval và candidate generation.
5. Không dùng heuristic regex làm cơ chế refinement chính. Toàn bộ semantic analysis và decomposition dùng LLM.

---

# 2. Tư duy kiến trúc mới

Module hiện tại chủ yếu làm:

```text
Question
   ↓
Regex cleaning
   ↓
Keyword extraction
   ↓
Target attribute
```

Cách mới:

```text
User Question
      │
      ▼
      LLM
      │
      ├───────────────┐
      │               │
      ▼               ▼
Question Analysis   Query Decomposition
      │               │
      │               ▼
      │          Sub-queries
      │               │
      ▼               ▼
Keywords / Intent   Retrieval Queries
Entities / Action        │
Modality                 │
      │                   │
      ├─────────┬─────────┤
      ▼         ▼         ▼
     OCR       ASR       OD / Visual
      │         │         │
      └─────────┼─────────┘
                ▼
             Retrieval
                ▼
           Candidate Pool
                ▼
             Reranking
                ▼
               VQA
```

Điểm quan trọng:

> **Keyword extraction và query decomposition là hai nhiệm vụ khác nhau.**

Keyword trả lời:

> "Trong câu hỏi có những thông tin/đối tượng nào quan trọng?"

Sub-query trả lời:

> "Tôi phải tìm những gì trong database để giải quyết câu hỏi?"

---

# 3. Chuẩn hóa English

Không cần duy trì:

```text
visual_query_vi
visual_query_en
keywords_vi
keywords_en
```

Thay vào đó:

```text
original_query
original_language
canonical_question
```

Trong đó:

- `original_query`: câu hỏi gốc của user.
- `original_language`: ngôn ngữ gốc.
- `canonical_question`: câu hỏi đã được LLM hiểu và biểu diễn bằng English.

Ví dụ:

```text
Original:
"Người đàn ông mặc áo đỏ đang đưa chai nước cho ai?"

Canonical:
"Who is the man wearing a red shirt giving a water bottle to?"
```

Toàn bộ downstream semantic retrieval dùng English.

---

# 4. Hai hướng output chính

## 4.1. Hướng A — Question Analysis

Đây là output được tạo trực tiếp từ **question**.

Nó không nhất thiết là câu truy vấn để retrieval.

Nó dùng để:

- xác định intent;
- xác định modality;
- lấy keyword/entity;
- xác định object;
- xác định action;
- xác định attribute;
- chọn OCR/ASR/OD/Visual/Metadata;
- hỗ trợ router/filter.

Ví dụ:

```text
Question:
"Người đàn ông mặc áo đỏ đang cầm bao nhiêu chai nước?"
```

LLM có thể trả:

```json
{
  "intent": "object_count",
  "modalities": ["visual", "od"],
  "entities": [
    "man",
    "water bottle"
  ],
  "attributes": [
    "red shirt"
  ],
  "actions": [
    "holding"
  ],
  "keywords": [
    "man",
    "red shirt",
    "water bottle",
    "holding",
    "count"
  ]
}
```

### Mục đích

Ví dụ router có thể quyết định:

```text
object_count
      +
water bottle
      ↓
OD filter
```

hoặc:

```text
text-related question
      ↓
OCR
```

hoặc:

```text
speech-related question
      ↓
ASR
```

---

# 5. Hướng B — Context/Hint → Sub-query Decomposition

Context hoặc hint không nên chỉ được coi là keyword.

Nó được dùng để **phân rã câu hỏi phức tạp thành các retrieval sub-query độc lập**.

Ví dụ:

```text
Question:
"Người đàn ông mặc áo đen đưa chai nước cho người phụ nữ rồi đi vào cửa hàng ở đâu?"
```

Thay vì chỉ tạo một query dài:

```text
"man in black shirt giving water bottle to woman then entering store"
```

LLM decomposition thành:

```text
Sub-query 1:
"man wearing black shirt"

Sub-query 2:
"man giving water bottle to woman"

Sub-query 3:
"woman receiving water bottle from man"

Sub-query 4:
"man entering a store"
```

Mỗi sub-query có thể được retrieval độc lập.

```text
Q1 ──→ Visual Retrieval ──→ Candidates
Q2 ──→ Visual Retrieval ──→ Candidates
Q3 ──→ Visual Retrieval ──→ Candidates
Q4 ──→ Visual Retrieval ──→ Candidates
                              │
                              ▼
                            Fusion
                              │
                              ▼
                         Reranking
```

---

# 6. Context/Hint có vai trò gì?

Context/hint nên được dùng để giải quyết những thành phần mà question tự nó không đủ thông tin.

Ví dụ:

```text
Context:
"A man wearing a black shirt enters a store and talks to a woman."

Question:
"What does he do before entering the store?"
```

LLM cần resolve:

```text
"he" → man wearing black shirt
```

Sau đó tạo sub-query:

```text
"man in black shirt before entering store"
```

Hoặc decomposition:

```text
Target event:
"man entering store"

Previous event:
"what the man does before entering store"
```

Nếu context đã chứa candidate event/description, nó có thể được dùng để tạo retrieval queries có tính định hướng cao hơn.

---

# 7. Query decomposition không có nghĩa là chia mọi câu hỏi

LLM phải quyết định:

```text
needs_decomposition = true / false
```

### Query đơn giản

```text
"red car"
```

Output:

```json
{
  "needs_decomposition": false,
  "sub_queries": [
    "red car"
  ]
}
```

Không cần tạo nhiều query vô ích.

### Query quan hệ

```text
"man talking to woman"
```

Có thể:

```json
{
  "needs_decomposition": true,
  "sub_queries": [
    "man talking",
    "woman talking",
    "man talking to woman"
  ]
}
```

### Query phức tạp

```text
"Người đàn ông mặc áo đen đưa chai nước cho người phụ nữ rồi đi vào cửa hàng."
```

Có thể:

```json
{
  "needs_decomposition": true,
  "sub_queries": [
    "man wearing black shirt",
    "man giving water bottle to woman",
    "woman receiving water bottle from man",
    "man entering store"
  ]
}
```

---

# 8. Sub-query phải giữ semantic anchor

Không được chia quá nhỏ làm mất ngữ nghĩa.

Không nên:

```text
"man"
"black"
"shirt"
"bottle"
"woman"
"store"
```

Đây chỉ là keywords.

Nên:

```text
"man wearing black shirt"
"man giving water bottle to woman"
"man entering store"
```

Mỗi sub-query phải đủ thông tin để một retriever có thể tìm candidate.

Nguyên tắc:

> **Keyword là atomic search signal; sub-query là semantic retrieval unit.**

---

# 9. Schema mới đề xuất

```python
class QueryAnalysis(BaseModel):
    original_query: str
    original_language: str

    canonical_question: str

    intent: str

    modalities: List[str]

    keywords: List[str]

    entities: List[str]

    attributes: List[str]

    actions: List[str]

    temporal_relations: List[str]


class QueryDecomposition(BaseModel):
    needs_decomposition: bool

    sub_queries: List[str]

    relations: List[str]


class RefinedQuery(BaseModel):
    analysis: QueryAnalysis

    decomposition: QueryDecomposition
```

Có thể mở rộng sau nếu cần:

```python
class RetrievalQuery(BaseModel):
    query: str
    modality: str
    purpose: str
```

Ví dụ:

```json
{
  "query": "man wearing black shirt",
  "modality": "visual",
  "purpose": "identify_person"
}
```

---

# 10. Modality nên được xác định riêng

Không nên dùng một field `target_attribute` để chứa tất cả.

Nên:

```text
modalities:
[
  "visual",
  "ocr",
  "asr",
  "od",
  "metadata"
]
```

Ví dụ:

### OCR

```text
Question:
"What does the sign say?"
```

```json
{
  "intent": "text_reading",
  "modalities": ["ocr"],
  "keywords": ["sign", "text"]
}
```

### ASR

```text
"What does the man say?"
```

```json
{
  "intent": "speech_content",
  "modalities": ["asr"],
  "keywords": ["man", "speech"]
}
```

### OD

```text
"How many cars are there?"
```

```json
{
  "intent": "object_count",
  "modalities": ["visual", "od"],
  "keywords": ["car", "count"]
}
```

### Visual

```text
"What color is the man's shirt?"
```

```json
{
  "intent": "attribute",
  "modalities": ["visual", "od"],
  "keywords": ["man", "shirt", "color"]
}
```

---

# 11. LLM Prompt nên yêu cầu hai nhiệm vụ riêng

Prompt không nên chỉ:

```text
"Extract keywords"
```

Mà phải yêu cầu:

```text
TASK 1:
Analyze the question and extract retrieval-relevant semantic information.

TASK 2:
Determine whether the question requires decomposition.

TASK 3:
If necessary, decompose the question into independent semantic
retrieval sub-queries.

TASK 4:
Use context/hints to resolve references and generate better sub-queries.

All retrieval queries must be in English.
```

---

# 12. Output JSON đề xuất

```json
{
  "original_language": "vi",

  "canonical_question":
    "Where did the man wearing a black shirt give the water bottle to the woman?",

  "intent": "location",

  "modalities": [
    "visual",
    "od"
  ],

  "keywords": [
    "man",
    "black shirt",
    "water bottle",
    "woman",
    "give",
    "location"
  ],

  "entities": [
    "man",
    "woman",
    "water bottle"
  ],

  "attributes": [
    "black shirt"
  ],

  "actions": [
    "give",
    "receive"
  ],

  "temporal_relations": [],

  "needs_decomposition": true,

  "sub_queries": [
    "man wearing black shirt",
    "man giving water bottle to woman",
    "woman receiving water bottle from man"
  ]
}
```

---

# 13. Context-aware example

Input:

```text
Context:
"A man wearing a black shirt enters a store. He approaches a woman
standing near the counter."

Question:
"What does he do before entering the store?"
```

Output:

```json
{
  "canonical_question":
    "What does the man wearing a black shirt do before entering the store?",

  "intent": "temporal_event",

  "modalities": [
    "visual"
  ],

  "keywords": [
    "man",
    "black shirt",
    "enter store",
    "before"
  ],

  "entities": [
    "man",
    "store"
  ],

  "actions": [
    "enter"
  ],

  "temporal_relations": [
    "before"
  ],

  "needs_decomposition": true,

  "sub_queries": [
    "man wearing black shirt",
    "man entering store",
    "event before man enters store"
  ]
}
```

Ở đây context có nhiệm vụ **ground reference** (`he`) và cung cấp anchor cho decomposition.

---

# 14. Retrieval flow cuối cùng

Pipeline nên trở thành:

```text
                     User
                      │
                      ▼
                   Question
                      │
                      ▼
                ┌───────────┐
                │    LLM    │
                │ Query     │
                │ Analysis  │
                └─────┬─────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Question Analysis   Decomposition
             │                 │
             │                 ▼
             │             Sub-queries
             │                 │
             │       ┌─────────┼─────────┐
             │       ▼         ▼         ▼
             │     Visual     OCR       ASR
             │       │         │         │
             │       └─────────┼─────────┘
             │                 ▼
             │            Candidate Pool
             │                 │
             └─────────────────┤
                               ▼
                            Fusion
                               │
                               ▼
                          Reranking
                               │
                               ▼
                            VQA/LLM
                               │
                               ▼
                             Answer
```

---

# 15. Điểm cần lưu ý khi implement

### Không cần heuristic refinement

Có thể bỏ:

```python
VI_STOP_PHRASES
EN_STOP_PHRASES
refine_heuristic()
```

vì semantic interpretation/decomposition chuyển hoàn toàn sang LLM.

Tuy nhiên vẫn nên giữ các xử lý kỹ thuật đơn giản như:

```text
JSON validation
schema validation
fallback
timeout
retry
length limit
```

Đây là infrastructure, không phải semantic heuristic.

### LLM failure

Nếu LLM trả JSON lỗi:

```text
LLM
 ↓
JSON parse failed
 ↓
Retry / constrained generation
 ↓
Nếu vẫn fail → dùng original query
```

Không nên cố gắng dùng regex để "giả lập" decomposition.

---

# 16. Nguyên tắc quan trọng nhất

Module mới cần tuân theo nguyên tắc:

```text
Question
   │
   ├──► Analysis
   │       ├── intent
   │       ├── modality
   │       ├── keywords
   │       ├── entities
   │       ├── attributes
   │       └── actions
   │
   └──► Decomposition
           ├── sub-query 1
           ├── sub-query 2
           ├── sub-query 3
           └── ...
```

**Analysis và Decomposition không thay thế nhau.**

- `keywords` → giúp Router quyết định **nên dùng OCR/ASR/OD/Visual/Metadata nào**.
- `sub_queries` → giúp Retrieval quyết định **cụ thể phải tìm những semantic event/entity nào**.

Đây là hai hướng độc lập nhưng bổ trợ cho nhau.

---

# 17. Kết luận triển khai

File `query_refiner.py` mới nên được thiết kế lại theo hướng:

```text
LLM Query Analyzer
        +
LLM Query Decomposer
        =
Query Refinement Module
```

Không còn trọng tâm là bilingual keyword extraction.

Thay vào đó:

```text
Vietnamese / English Question
             │
             ▼
       LLM Understanding
             │
             ├── Canonical English Question
             │
             ├── Keywords
             │      └── OCR / ASR / OD / Metadata filters
             │
             └── Sub-query Decomposition
                    └── Retrieval
```

Đây là kiến trúc phù hợp hơn với hệ thống VQA/AVS hiện tại vì **LLM không trực tiếp trả lời câu hỏi ở bước này**. Nó chỉ biến một câu hỏi phức tạp thành **cấu trúc thông tin + các retrieval units**, sau đó các tầng Retrieval/Reranking/VQA mới xử lý tiếp.
