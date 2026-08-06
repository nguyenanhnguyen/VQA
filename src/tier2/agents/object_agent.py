import os
import json
from typing import List, Dict, Any
from ...utils.text_utils import normalize_text
from ...utils.file_utils import read_jsonl
from ...config.settings import settings
from .base import BaseAgent
from ...utils.logging import logger

class ObjectAgent(BaseAgent):
    def __init__(self, top_k: int = 200):
        self.top_k = top_k
        self.object_dir = settings.OBJECT_JSON_DIR
        self.shot_file = settings.SHOT_JSONL
    
    def get_source_name(self) -> str:
        return "object"
    
    def search(self, query: str, video_filter: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        # Query chứa tên object cần tìm
        # Heuristic: tìm từ khóa object trong query
        # TODO: thực tế cần mapping từ query -> object category
        # Tạm thời lấy danh sách object từ file objects.json (nếu có)
        candidates = []
        # Placeholder: cần có dữ liệu object cho từng keyframe
        logger.warning("ObjectAgent not fully implemented")
        return candidates