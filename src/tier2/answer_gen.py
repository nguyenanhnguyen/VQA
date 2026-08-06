from typing import List, Dict, Any, Optional
from PIL import Image
from ..models.vlm_loader import generate_answer
from ..utils.file_utils import load_image
from ..config.settings import settings
from ..utils.logging import logger
import os

def generate_answer_for_candidate(question: str, candidate: Dict[str, Any], num_frames: int = 5) -> str:
    """
    Generate answer for a candidate using VLM with surrounding frames
    """
    video_id = candidate["video_id"]
    frame_id = candidate["frame_id"]
    
    # Get keyframe path
    keyframe_dir = settings.KEYFRAME_DIR
    # Giả định keyframe được lưu với tên số thứ tự, cần mapping frame_id -> file
    # Tạm thời tìm file gần nhất (có thể không đúng)
    # TODO: implement proper frame mapping
    frame_file = os.path.join(keyframe_dir, video_id, f"{frame_id:04d}.jpg")
    if not os.path.exists(frame_file):
        # Try to find any image in that directory
        frame_dir = os.path.join(keyframe_dir, video_id)
        if os.path.exists(frame_dir):
            files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
            if files:
                # take the first file as fallback
                frame_file = os.path.join(frame_dir, files[0])
            else:
                return "Không tìm thấy hình ảnh cho ứng viên này."
        else:
            return "Không tìm thấy hình ảnh cho ứng viên này."
    
    try:
        image = load_image(frame_file)
        answer = generate_answer(question, [image], max_new_tokens=100)
        return answer
    except Exception as e:
        logger.error(f"VLM answer generation failed: {e}")
        return "Lỗi khi sinh câu trả lời."