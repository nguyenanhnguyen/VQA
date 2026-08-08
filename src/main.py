"""
src/main.py
============
Main API Service - VQA Module (Cascade Redesign Architecture)
"""

import time
from fastapi import FastAPI
from typing import List, Dict, Any

from .schemas.request import VQARequest
from .schemas.response import VQAResponse, Candidate, Evidence, VQAError
from .router import get_router
from .tier0 import tier0_search, synthesize_tier0_answer
from .tier2 import iterative_retrieval, generate_answer_for_candidate, light_verify_candidate
from .confidence import evaluate_gate_b_confidence
from .query_refiner import get_query_refiner
from .utils.logging import logger
from .utils.metrics import metrics
from .config.settings import settings, apply_cli_overrides

# Tự động áp dụng CLI overrides
apply_cli_overrides()

app = FastAPI(title="VQA Module", version="2.1.0")

# Router khởi tạo 1 lần lúc import module
router = get_router()


@app.get("/health")
async def health():
    return {"status": "ok"}


def _build_evidence(candidate: Dict[str, Any]) -> Evidence:
    """Chuyển 1 candidate dict thành đúng Evidence schema."""
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

    tier_attempted = "tier0"

    try:
        # 1. LLM Query Understanding & Query Decomposition
        refiner = get_query_refiner()
        refined_query = refiner.refine(request.question, context_hints=request.context_hints)

        # 2. Gate A Router Classification (Rule-based Heuristic constraint maintained)
        routing = router.route(request.question, refined_query=refined_query)
        tier_attempted = routing.tier if request.tier == "auto" else request.tier
        logger.info(f"Gate A Route decision: {tier_attempted} ({routing.reason})")
        metrics.record("tier", tier_attempted)

        answer: str = None
        confidence: float = 0.0

        if tier_attempted == "tier0":
            # 3. Tier 0 Multi-Agent Retrieval with Query Decomposition
            candidates = tier0_search(
                question=request.question,
                max_results=request.max_results,
                video_filter=request.video_filter,
                refined_query=refined_query
            )

            if candidates:
                top = candidates[0]
                # Fast Text-LLM Fact Synthesis (No VLM!)
                answer = synthesize_tier0_answer(request.question, top, refined_query=refined_query)

                # 4. Gate B Evidence-level Confidence Evaluator
                gate_b = evaluate_gate_b_confidence(request.question, candidates, refined_query=refined_query)
                confidence = gate_b.confidence_score

                # Escalation check
                if gate_b.should_escalate:
                    logger.info(f"Gate B triggered escalation to Tier 2 (confidence={confidence:.2f})")
                    tier_attempted = "tier2_escalated"

                    # 5. Tier 2 Stage 1: Light Verify (1-frame fast VLM check)
                    verified, v_answer, v_conf = light_verify_candidate(request.question, top, answer)

                    if verified:
                        logger.info("LightVerify confirmed Tier 0 answer.")
                        answer = v_answer
                        confidence = v_conf
                    else:
                        logger.info("LightVerify unsure. Falling back to Full Agentic Pipeline.")
                        result = iterative_retrieval(
                            question=request.question,
                            context_hints=request.context_hints,
                            max_iterations=3,
                            refined_query=refined_query
                        )
                        tier2_candidates = result.get("candidates", [])
                        if tier2_candidates:
                            candidates = tier2_candidates
                            answer = generate_answer_for_candidate(request.question, candidates[0])
                            confidence = result.get("confidence", 0.5)

        else:
            # Direct Tier 2: Full Agentic Pipeline with Query Decomposition
            result = iterative_retrieval(
                question=request.question,
                context_hints=request.context_hints,
                max_iterations=3,
                refined_query=refined_query
            )
            candidates = result.get("candidates", [])
            if candidates:
                top = candidates[0]
                answer = generate_answer_for_candidate(request.question, top)
                confidence = result.get("confidence", 0.5)
            else:
                answer = None
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
