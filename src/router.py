"""
src/router.py
===============
Gate A Router (Cascade Redesign):
Phân loại câu hỏi tại Gate A thành:
  - `tier0`: Có thể trả lời từ dữ liệu cấu trúc (OCR text, ASR speech, Metadata, Object presence/count).
  - `tier2`: Cần suy luận thị giác/không gian phức tạp (Spatial relations, complex action recognition, mood/emotion).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from .query_refiner import RefinedQuery


@dataclass
class RoutingDecision:
    tier: str  # "tier0" | "tier2"
    complexity_score: float
    reason: str


# Từ khóa cho câu hỏi cần suy luận thị giác/không gian phức tạp -> Tier 2
VISUAL_REASONING_MARKERS = [
    r"\bbên trái\b", r"\bbên phải\b", r"\bở giữa\b", r"\bphía trên\b", r"\bphía dưới\b",
    r"\bđang làm gì\b", r"\bhành động\b", r"\bcảm xúc\b", r"\bkhông gian\b",
    r"\bleft of\b", r"\bright of\b", r"\babove\b", r"\bbelow\b", r"\bdoing what\b",
    r"\baction\b", r"\bmood\b", r"\brelationship\b"
]

# Từ khóa cho câu hỏi trả lời được từ dữ liệu cấu trúc (OCR/ASR/Metadata/Object) -> Tier 0
STRUCTURED_EVIDENCE_MARKERS = [
    r"\bchữ\b", r"\bbiển báo\b", r"\bdòng chữ\b", r"\bviết\b", r"\btext\b", r"\breads\b",
    r"\bnói gì\b", r"\blời thoại\b", r"\bphát biểu\b", r"\bspeech\b", r"\bsaid\b",
    r"\btiêu đề\b", r"\btên kênh\b", r"\bngày đăng\b", r"\btitle\b", r"\bchannel\b",
    r"\bbao nhiêu\b", r"\bmấy\b", r"\bhow many\b"
]


class BaseRouter(ABC):
    @abstractmethod
    def route(self, question: str, refined_query: Optional[RefinedQuery] = None) -> RoutingDecision:
        ...


class GateARouter(BaseRouter):
    """Gate A Router: Rule-based heuristic phân loại dựa trên intent & keywords."""

    def route(self, question: str, refined_query: Optional[RefinedQuery] = None) -> RoutingDecision:
        q_lower = question.lower()

        # 1. Nếu có refined_query, ưu tiên target_attribute
        if refined_query:
            attr = refined_query.target_attribute
            if attr in ["text_ocr", "speech_asr", "metadata", "object_count"]:
                return RoutingDecision(
                    tier="tier0",
                    complexity_score=0.2,
                    reason=f"Refined intent '{attr}' is structured evidence answerable."
                )
            elif attr in ["action"]:
                return RoutingDecision(
                    tier="tier2",
                    complexity_score=0.8,
                    reason=f"Refined intent '{attr}' requires visual action reasoning."
                )

        # 2. Heuristic check trên từ khóa
        vis_hits = sum(1 for pat in VISUAL_REASONING_MARKERS if re.search(pat, q_lower))
        struct_hits = sum(1 for pat in STRUCTURED_EVIDENCE_MARKERS if re.search(pat, q_lower))

        score = min(1.0, 0.4 * vis_hits - 0.3 * struct_hits + 0.3)

        if vis_hits > struct_hits:
            tier = "tier2"
            reason = f"Visual reasoning markers ({vis_hits}) exceed structured markers ({struct_hits})"
        elif struct_hits > 0:
            tier = "tier0"
            reason = f"Structured evidence markers ({struct_hits}) present"
        else:
            tier = "tier0" if score < 0.6 else "tier2"
            reason = f"Default routing threshold check score={score:.2f}"

        return RoutingDecision(tier=tier, complexity_score=score, reason=reason)


def get_router() -> BaseRouter:
    """Factory return Gate A Router."""
    return GateARouter()
