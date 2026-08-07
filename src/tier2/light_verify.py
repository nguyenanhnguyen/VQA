"""
src/tier2/light_verify.py
===========================
Light Verify Stage (Cascade Redesign):
Thực hiện 1 lệnh kiểm tra VLM nhanh (1 khung hình) để xác minh xem câu trả lời
từ Tier 0 có khớp với bức ảnh keyframe hay không trước khi quyết định có chạy
toàn bộ 5-agent iterative planner pipeline hay không.
"""

import os
import logging
from typing import Dict, Any, Tuple
from ..utils.file_utils import load_image
from ..models.vlm_loader import generate_answer
from ..config.settings import settings

logger = logging.getLogger(__name__)


def light_verify_candidate(
    question: str,
    candidate: Dict[str, Any],
    candidate_answer: str
) -> Tuple[bool, str, float]:
    """
    Light Verification stage:
    Returns (verified: bool, final_answer: str, confidence: float)
    """
    video_id = candidate.get("video_id", "")
    frame_id = candidate.get("frame_id", 0)

    keyframe_dir = settings.KEYFRAME_DIR
    frame_file = os.path.join(keyframe_dir, video_id, f"{frame_id:04d}.jpg")

    if not os.path.exists(frame_file):
        frame_dir = os.path.join(keyframe_dir, video_id)
        if os.path.exists(frame_dir):
            files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
            if files:
                frame_file = os.path.join(frame_dir, files[0])

    if not os.path.exists(frame_file):
        logger.warning(f"LightVerify: Keyframe image not found for video {video_id}, frame {frame_id}")
        return False, candidate_answer, 0.0

    try:
        image = load_image(frame_file)
        verify_prompt = (
            f"Câu hỏi: {question}\n"
            f"Câu trả lời đề xuất: {candidate_answer}\n"
            f"Dựa vào bức ảnh này, câu trả lời đề xuất trên có ĐÚNG không? "
            f"Trả lời 'ĐÚNG' hoặc 'SAI' và nêu lý do ngắn gọn trong 1 câu."
        )

        verification = generate_answer(verify_prompt, [image], max_new_tokens=60).strip()
        logger.info(f"LightVerify output: '{verification}'")

        upper_v = verification.upper()
        if "ĐÚNG" in upper_v or "YES" in upper_v or "CORRECT" in upper_v:
            return True, candidate_answer, 0.85
        else:
            return False, verification, 0.3

    except Exception as e:
        logger.error(f"LightVerify VLM call failed: {e}")
        return False, candidate_answer, 0.0
