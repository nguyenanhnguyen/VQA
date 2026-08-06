import time
from fastapi import FastAPI
from typing import List, Dict, Any

from .schemas.request import VQARequest
from .schemas.response import VQAResponse, Candidate, Evidence, VQAError
from .router import get_router
from .tier0 import tier0_search
from .tier2 import iterative_retrieval, generate_answer_for_candidate
from .utils.logging import logger
from .utils.metrics import metrics
from .config.settings import settings

app = FastAPI(title="VQA Module", version="1.0.0")

# Router khởi tạo 1 lần lúc import module (HeuristicRouter không cần kết nối
# gì cả nên an toàn để tạo eager ở đây, khác Milvus/model phải lazy).
router = get_router()


@app.get("/health")
async def health():
    return {"status": "ok"}


def _build_evidence(candidate: Dict[str, Any]) -> Evidence:
    """Chuyển 1 candidate dict (từ tier0/tier2) thành đúng Evidence schema.
    Evidence.type là field bắt buộc -- suy ra từ 'source' nếu agent có gắn,
    mặc định 'fusion' vì candidate thường đã qua RRF gộp nhiều nguồn."""
    return Evidence(
        type=candidate.get("source", "fusion"),
        description=candidate.get("description"),
        ocr_text=candidate.get("ocr_text"),
        asr_text=candidate.get("asr_text"),
    )


@app.post("/vqa/query", response_model=VQAResponse)
async def vqa_query(request: VQARequest):
    start_time = time.time()
    logger.info(f"Received request {request.request_id}: {request.question[:100]}...")

    # tier mặc định để dùng luôn cả trong except block nếu crash trước khi router chạy xong
    tier_attempted = "tier0"

    try:
        tier_attempted = router.route(request.question).tier if request.tier == "auto" else request.tier
        logger.info(f"Route: {tier_attempted}")
        metrics.record("tier", tier_attempted)

        answer: str = None
        confidence: float = 0.0
        evidence_records: List[dict] = []

        if tier_attempted == "tier0":
            candidates = tier0_search(
                question=request.question,
                max_results=request.max_results,
                video_filter=request.video_filter,
            )
            if candidates:
                top = candidates[0]
                answer = generate_answer_for_candidate(request.question, top)
                confidence = top.get("score", 0.5)
        else:
            result = iterative_retrieval(
                question=request.question,
                context_hints=request.context_hints,
                max_iterations=3,
            )
            candidates = result.get("candidates", [])
            evidence_records = result.get("evidence", [])
            if candidates:
                top = candidates[0]
                answer = generate_answer_for_candidate(request.question, top)
                confidence = result.get("confidence", 0.5)
            else:
                answer = None  # KHÔNG bịa câu trả lời khi không có candidate (VQA-T12 fallback)
                confidence = 0.0

        response_candidates = []
        for idx, c in enumerate(candidates[: request.max_results], start=1):
            response_candidates.append(Candidate(
                rank=idx,
                video_id=c.get("video_id", ""),
                frame_id=c.get("frame_id", 0),
                score=c.get("score", 0.0),
                confidence=c.get("rerank_score", c.get("score", 0.0)),
                answer=answer if idx == 1 else None,
                evidence=_build_evidence(c),
            ))

        elapsed_ms = int((time.time() - start_time) * 1000)
        metrics.record("time_latency_ms", elapsed_ms)

        error_obj = None
        if not response_candidates:
            error_obj = VQAError(code="NO_EVIDENCE", message="Không tìm đủ bằng chứng để trả lời câu hỏi này.")

        return VQAResponse(
            request_id=request.request_id,
            tier_used=tier_attempted,
            model_version=settings.MODEL_VERSION_TAG,
            candidates=response_candidates,
            final_answer=answer,
            final_confidence=confidence,
            latency_ms=elapsed_ms,
            error=error_obj,
        )

    except Exception as e:
        logger.error(f"Error processing request {request.request_id}: {e}")
        elapsed_ms = int((time.time() - start_time) * 1000)
        return VQAResponse(
            request_id=request.request_id,
            tier_used=tier_attempted,
            model_version=settings.MODEL_VERSION_TAG,
            candidates=[],
            final_answer=None,
            final_confidence=None,
            latency_ms=elapsed_ms,
            error=VQAError(code="MODEL_ERROR", message=str(e)),
        )
