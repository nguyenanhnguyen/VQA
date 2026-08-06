"""
schemas/request.py
====================
Pydantic model cho input của VQA service — khớp CHÍNH XÁC với contract
đã định nghĩa trong vqa_module.txt (không tự ý thêm/bớt field).

Đây là 1 phần của "giao thức" (protocol) VQA-T09 yêu cầu: mọi request đi vào
router.py hay gọi trực tiếp API phải theo đúng schema này để Router/UI dùng
được mà không cần biết logic bên trong.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class FrameRange(BaseModel):
    start: int = Field(..., ge=0, description="Frame bắt đầu (inclusive)")
    end: int = Field(..., ge=0, description="Frame kết thúc (inclusive)")


class VQARequest(BaseModel):
    request_id: str = Field(..., description="ID duy nhất cho mỗi request, dùng để trace log/latency.")
    question: str = Field(..., min_length=1, description="Câu hỏi ngôn ngữ tự nhiên (tiếng Việt hoặc Anh).")
    context_hints: List[str] = Field(
        default_factory=list,
        description="Chỉ dùng ở vòng chung kết (Conversational KIS) — các gợi ý bổ sung từ hội thoại trước.",
    )
    video_filter: Optional[List[str]] = Field(
        default=None,
        description="Danh sách video_id cần giới hạn tìm kiếm. None = tìm toàn bộ database.",
    )
    frame_range: Optional[FrameRange] = Field(
        default=None,
        description="Giới hạn phạm vi frame trong video (chỉ có ý nghĩa khi video_filter chỉ định đúng 1 video).",
    )
    max_results: int = Field(default=100, ge=1, le=1000, description="Số candidate tối đa trả về.")
    tier: Literal["auto", "tier0", "tier2"] = Field(
        default="auto",
        description="'auto' để router.py tự quyết định; ép cứng 'tier0'/'tier2' phục vụ test/debug.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "vqa-001",
                "question": "Người đàn ông mặc áo đỏ đang nói chuyện điện thoại ở đâu?",
                "context_hints": [],
                "video_filter": None,
                "frame_range": None,
                "max_results": 100,
                "tier": "auto",
            }
        }
