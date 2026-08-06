"""
src/router.py
===============
Adaptive Router (theo Adaptive-RAG) — quyết định câu hỏi đi Tier 0 (nhanh,
1 bước retrieval) hay Tier 2 (Agentic, nhiều bước).

QUYẾT ĐỊNH THIẾT KẾ: viết 2 lớp router, chọn qua config, để hệ thống CHẠY
ĐƯỢC ngay cả khi chưa có API key LLM (heuristic), đồng thời cho phép nâng
cấp lên LLM-based khi đã có key — không block việc build/test các phần khác.

  - HeuristicRouter (mặc định): rule-based, không cần LLM, không tốn latency.
  - LLMRouter (tuỳ chọn): dùng model nhỏ (Gemini-2.5-Flash-Lite) để phân loại,
    CẦN GEMINI_API_KEY. Bật bằng ROUTER_LLM_MODEL trong .env + gọi
    LLMRouter thay vì HeuristicRouter trong main.py.

CHƯA CÓ: ngưỡng độ phức tạp (ROUTER_COMPLEXITY_THRESHOLD) hiện là placeholder
0.5, phải hiệu chỉnh lại bằng bộ 30 câu hỏi thật ở VQA-T07.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config.settings import settings


@dataclass
class RoutingDecision:
    tier: str  # "tier0" | "tier2"
    complexity_score: float
    reason: str


# Các dấu hiệu cho thấy câu hỏi cần suy luận đa bước (multi-hop) -> Tier 2
MULTI_HOP_MARKERS = [
    r"\bsau đó\b", r"\btrước khi\b", r"\btiếp theo\b", r"\bkế tiếp\b",
    r"\bvà sau\b", r"\bbao nhiêu\b", r"\bđếm\b", r"\bai là người\b",
    r"\bthen\b", r"\bafter\b", r"\bbefore\b", r"\bhow many\b", r"\bcount\b",
]


class BaseRouter(ABC):
    @abstractmethod
    def route(self, question: str) -> RoutingDecision:
        ...


class HeuristicRouter(BaseRouter):
    """Rule-based: không cần LLM, chạy được ngay không phụ thuộc API key."""

    def route(self, question: str) -> RoutingDecision:
        q_lower = question.lower()

        marker_hits = sum(1 for pat in MULTI_HOP_MARKERS if re.search(pat, q_lower))
        word_count = len(question.split())

        # Heuristic thô: câu hỏi dài + có marker suy luận nhiều bước -> phức tạp.
        # (Đây là baseline tạm thời, CẦN thay bằng số liệu thật từ benchmark)
        score = min(1.0, 0.15 * marker_hits + 0.02 * word_count)

        tier = "tier2" if score >= settings.ROUTER_COMPLEXITY_THRESHOLD else "tier0"
        reason = f"marker_hits={marker_hits}, word_count={word_count}, score={score:.2f}"
        return RoutingDecision(tier=tier, complexity_score=score, reason=reason)


class LLMRouter(BaseRouter):
    """
    Dùng LLM nhỏ để phân loại độ phức tạp câu hỏi.
    YÊU CẦU: GEMINI_API_KEY (hoặc provider khác) đã cấu hình trong .env.
    Hiện để interface sẵn, implementation gọi API thật cần xác nhận
    provider/SDK chính thức nhóm dùng (xem câu hỏi cuối cuộc trò chuyện).
    """

    def __init__(self):
        raise NotImplementedError(
            "LLMRouter cần xác nhận: (1) provider LLM thật (OpenAI/Gemini/nội bộ), "
            "(2) SDK/cách gọi API đã có trong hệ thống chưa. Xem phần câu hỏi context."
        )

    def route(self, question: str) -> RoutingDecision:
        raise NotImplementedError


def get_router() -> BaseRouter:
    """Factory — main.py gọi hàm này thay vì new trực tiếp, dễ đổi router sau."""
    return HeuristicRouter()
