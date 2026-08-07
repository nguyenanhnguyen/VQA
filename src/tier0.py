"""
src/tier0.py
===============
Tier 0 Search & Fact Synthesizer (Cascade Redesign):
- Sử dụng CLIP search + Multi-Agent retrieval.
- Dùng Text-only LLM tổng hợp câu trả lời từ các bằng chứng (transcripts, OCR, metadata).
- KHÔNG gọi VLM (Vision Language Model) xử lý ảnh -> Đảm bảo độ trễ siêu thấp (<1s).
"""

from typing import List, Dict, Any, Optional
from .database.milvus_client import get_milvus_client
from .models.clip_loader import get_clip_loader
from .tier2.fusion import reciprocal_rank_fusion
from .tier2.agents.asr_agent import ASRAgent
from .tier2.agents.metadata_agent import MetadataAgent
from .tier2.agents.ocr_agent import OCRAgent
from .tier2.agents.object_agent import ObjectAgent
from .models.llm_loader import call_llm
from .query_refiner import RefinedQuery, get_query_refiner
from .utils.logging import logger


def tier0_search(
    question: str,
    max_results: int = 100,
    video_filter: Optional[List[str]] = None,
    refined_query: Optional[RefinedQuery] = None
) -> List[Dict[str, Any]]:
    """
    Thực hiện truy xuất đa nguồn cho Tier 0:
    1. Vision (CLIP vector search trong Milvus bằng refined visual_query)
    2. ASR (Speech transcripts bằng keywords)
    3. Metadata (Title/Description bằng keywords)
    4. OCR & Object agents
    Gộp điểm bằng Reciprocal Rank Fusion (RRF).
    """
    if not refined_query:
        refiner = get_query_refiner()
        refined_query = refiner.refine(question)

    logger.info(f"Tier0: searching for visual_query='{refined_query.visual_query_en}'")

    # 1. Vision agent (CLIP + Milvus)
    embedder = get_clip_loader()
    search_prompt = refined_query.visual_query_en or refined_query.visual_query_vi
    query_vector = embedder.encode_text(search_prompt)[0].tolist()

    expr = None
    if video_filter:
        video_list = ", ".join([f"'{v}'" for v in video_filter])
        expr = f"video_id in [{video_list}]"

    milvus = get_milvus_client()
    vision_hits = milvus.search(query_vector, top_k=max_results, expr=expr)

    vision_candidates = []
    for hit in vision_hits:
        vision_candidates.append({
            "video_id": hit.entity.get("video_id", ""),
            "frame_id": hit.entity.get("frame_id", 0),
            "score": hit.score,
            "source": "clip",
        })

    # 2. ASR agent search
    asr_agent = ASRAgent(top_k=50)
    asr_candidates = asr_agent.search(question, video_filter=video_filter)

    # 3. Metadata agent search
    meta_agent = MetadataAgent(top_k=50)
    meta_candidates = meta_agent.search(question, video_filter=video_filter)

    # 4. OCR & Object agents
    ocr_agent = OCRAgent(top_k=50)
    ocr_candidates = ocr_agent.search(question, video_filter=video_filter)

    obj_agent = ObjectAgent(top_k=50)
    obj_candidates = obj_agent.search(question, video_filter=video_filter)

    # Gộp danh sách ứng viên qua RRF Fusion
    ranked = reciprocal_rank_fusion([
        vision_candidates, asr_candidates, meta_candidates, ocr_candidates, obj_candidates
    ], top_k=max_results)

    return ranked


def synthesize_tier0_answer(question: str, top_candidate: Dict[str, Any], refined_query: Optional[RefinedQuery] = None) -> str:
    """
    Tổng hợp câu trả lời cho Tier 0 bằng Text-only LLM từ các facts đã thu thập.
    KHÔNG dùng VLM mã hóa ảnh -> Siêu nhanh.
    """
    evidence_text = []
    if top_candidate.get("asr_text"):
        evidence_text.append(f"Lời thoại ASR: {top_candidate['asr_text']}")
    if top_candidate.get("ocr_text"):
        evidence_text.append(f"Chữ OCR: {top_candidate['ocr_text']}")
    if top_candidate.get("description"):
        evidence_text.append(f"Mô tả/Metadata: {top_candidate['description']}")

    if not evidence_text:
        return f"Dựa trên dữ liệu video {top_candidate.get('video_id', '')}, khung hình {top_candidate.get('frame_id', 0)}."

    context_str = "\n".join(evidence_text)

    messages = [
        {
            "role": "system",
            "content": "Bạn là trợ lý VQA. Hãy trả lời câu hỏi của người dùng một cách ngắn gọn, chính xác bằng tiếng Việt dựa TRỰC TIẾP vào các dữ kiện thu thập được bên dưới. Nếu không đủ thông tin, hãy trả lời ngắn gọn những gì tìm thấy."
        },
        {
            "role": "user",
            "content": f"Dữ kiện thu thập được:\n{context_str}\n\nCâu hỏi: {question}\nTrả lời ngắn gọn:"
        }
    ]

    try:
        answer = call_llm(messages, temperature=0.0)
        if answer:
            return answer
    except Exception as e:
        logger.warning(f"Tier0 LLM fact synthesis fallback: {e}")

    return f"Đã tìm thấy bằng chứng khớp tại video {top_candidate.get('video_id', '')} khung hình {top_candidate.get('frame_id', 0)}: {context_str[:150]}"
