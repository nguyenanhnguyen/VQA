"""
schemas/response.py
=====================
Pydantic model cho output của VQA service — khớp CHÍNH XÁC với contract
trong vqa_module.txt, cộng thêm phần "chuẩn hóa answer/evidence" theo
đúng yêu cầu VQA-T09 (model_version, no-evidence flag, error chuẩn).
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    type: Literal["vision", "ocr", "asr", "object", "metadata", "fusion"] = Field(
        ..., description="Nguồn bằng chứng nào đóng góp chính cho candidate này."
    )
    description: Optional[str] = Field(default=None, description="Mô tả ngắn cảnh trong keyframe (từ caption/VLM).")
    ocr_text: Optional[str] = Field(default=None, description="Text OCR đã qua LLM sửa lỗi (nếu có).")
    asr_text: Optional[str] = Field(default=None, description="Transcript PhoWhisper của shot tương ứng (nếu có).")


class Candidate(BaseModel):
    rank: int = Field(..., ge=1)
    video_id: str
    frame_id: int
    score: float = Field(..., ge=0.0, le=1.0, description="Điểm truy xuất/rerank thô.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy answer_gen.py gán cho candidate này.")
    answer: Optional[str] = Field(default=None, description="None nếu candidate không đủ căn cứ để trả lời.")
    evidence: Evidence


class VQAError(BaseModel):
    code: Literal[
        "NO_EVIDENCE",      # không tìm đủ bằng chứng để trả lời (VQA-T12 fallback)
        "TIMEOUT",
        "RETRIEVAL_FAILED",
        "MODEL_ERROR",
        "INVALID_REQUEST",
    ]
    message: str


class VQAResponse(BaseModel):
    request_id: str
    tier_used: Literal["tier0", "tier2"]
    model_version: str = Field(..., description="Tag baseline đã đóng băng, vd 'vqa-baseline-v1.2' (VQA-T09).")
    candidates: List[Candidate] = Field(default_factory=list)
    final_answer: Optional[str] = Field(
        default=None, description="None nếu error.code == NO_EVIDENCE — KHÔNG được bịa answer khi thiếu bằng chứng."
    )
    final_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    latency_ms: int = Field(..., ge=0)
    error: Optional[VQAError] = Field(default=None, description="None nếu request thành công bình thường.")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "vqa-001",
                "tier_used": "tier2",
                "model_version": "vqa-baseline-v1.0",
                "candidates": [
                    {
                        "rank": 1,
                        "video_id": "L01_V001",
                        "frame_id": 1450,
                        "score": 0.92,
                        "confidence": 0.88,
                        "answer": "Trong công viên gần hồ nước",
                        "evidence": {
                            "type": "vision",
                            "description": "Khung hình người đàn ông áo đỏ ngồi trên ghế đá cạnh hồ",
                            "ocr_text": "",
                            "asr_text": "",
                        },
                    }
                ],
                "final_answer": "Trong công viên gần hồ nước",
                "final_confidence": 0.88,
                "latency_ms": 2345,
                "error": None,
            }
        }
