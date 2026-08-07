"""
src/confidence.py
===================
Gate B Evidence-level Confidence Evaluator (Cascade Redesign):
Đánh giá độ tin cậy của kết quả Tier 0 dựa trên bằng chứng thu thập được:
  1. Score Margin: Chênh lệch điểm giữa ứng viên top-1 và top-2.
  2. Agent Agreement: Số lượng agent độc lập (Vision, ASR, OCR, Metadata) cùng tìm ra 1 video/frame.
  3. Coverage Score: Đã có bằng chứng từ đúng agent chuyên trách cho dạng câu hỏi đó chưa.
Quyết định có leo thang (escalate) sang Tier 2 hay không.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from config.settings import settings
from .query_refiner import RefinedQuery

logger = logging.getLogger(__name__)


@dataclass
class GateBResult:
    confidence_score: float
    should_escalate: bool
    margin_score: float
    agreement_score: float
    coverage_score: float
    rationale: str


def evaluate_gate_b_confidence(
    question: str,
    candidates: List[Dict[str, Any]],
    refined_query: Optional[RefinedQuery] = None,
    threshold: Optional[float] = None
) -> GateBResult:
    if threshold is None:
        threshold = getattr(settings, "GATE_B_CONFIDENCE_THRESHOLD", 0.65)

    if not candidates:
        return GateBResult(
            confidence_score=0.0,
            should_escalate=True,
            margin_score=0.0,
            agreement_score=0.0,
            coverage_score=0.0,
            rationale="No candidates retrieved from Tier 0 search."
        )

    top1 = candidates[0]
    top1_video = top1.get("video_id", "")
    top1_score = top1.get("score", 0.0)

    # 1. Margin Score
    if len(candidates) > 1:
        top2_score = candidates[1].get("score", 0.0)
        # Normalize margin (RRF scores typically ~0.01 - 0.05)
        raw_margin = max(0.0, top1_score - top2_score)
        margin_score = min(1.0, raw_margin * 20.0)
    else:
        margin_score = 0.5

    # 2. Agent Agreement Score
    # Count distinct sources in top 10 candidates pointing to top1_video
    agreeing_sources = set()
    for c in candidates[:10]:
        if c.get("video_id") == top1_video:
            src = c.get("source", "fusion")
            agreeing_sources.add(src)

    # Max 4 sources (clip, asr, metadata, ocr/object)
    agreement_score = min(1.0, len(agreeing_sources) / 3.0)

    # 3. Coverage Score
    coverage_score = 0.5
    if refined_query:
        attr = refined_query.target_attribute
        if attr == "text_ocr" and top1.get("ocr_text"):
            coverage_score = 1.0
        elif attr == "speech_asr" and top1.get("asr_text"):
            coverage_score = 1.0
        elif attr == "metadata" and top1.get("description"):
            coverage_score = 1.0
        elif top1.get("source") == "clip":
            coverage_score = 0.7

    # Weighted Confidence Formula
    confidence = (0.4 * margin_score) + (0.4 * agreement_score) + (0.2 * coverage_score)
    confidence = round(confidence, 3)

    should_escalate = confidence < threshold

    rationale = (
        f"Gate B: conf={confidence:.3f} (thresh={threshold}), margin={margin_score:.2f}, "
        f"agreement={agreement_score:.2f} (sources={list(agreeing_sources)}), coverage={coverage_score:.2f}. "
        f"Escalate={should_escalate}"
    )
    logger.info(rationale)

    return GateBResult(
        confidence_score=confidence,
        should_escalate=should_escalate,
        margin_score=margin_score,
        agreement_score=agreement_score,
        coverage_score=coverage_score,
        rationale=rationale
    )
