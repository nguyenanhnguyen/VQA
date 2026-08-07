import os
import json
from typing import List, Dict, Any
from ...database.metadata_client import metadata_client
from ...utils.text_utils import normalize_text
from .base import BaseAgent
from ...utils.logging import logger
from ...config.settings import settings

class OCRAgent(BaseAgent):
    def __init__(self, top_k: int = 200):
        self.top_k = top_k
        self.ocr_dir = settings.OCR_DIR or settings.OBJECT_JSON_DIR
    
    def get_source_name(self) -> str:
        return "ocr"
    
    def search(self, query: str, video_filter: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        # Giả định có file OCR cho từng keyframe (có thể trong thư mục ocr)
        # Ở đây ta mô phỏng: đọc OCR từ file hoặc từ một index.
        # Thực tế cần có OCR index (có thể Elasticsearch hoặc cache)
        # Tạm thời trả về rỗng nếu chưa có dữ liệu
        # TODO: implement OCR search using precomputed OCR text
        candidates = []
        # Placeholder: nếu có OCR data, ta sẽ tìm kiếm ở đây
        logger.warning("OCRAgent not fully implemented")
        return candidates