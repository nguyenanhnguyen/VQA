import os
from typing import List, Dict, Any, Optional
from ...utils.text_utils import normalize_text
from ...utils.file_utils import read_jsonl
from ...config.settings import settings
from .base import BaseAgent
from ...utils.logging import logger
from ...query_refiner import RefinedQuery


class OCRAgent(BaseAgent):
    def __init__(self, top_k: int = 200):
        self.top_k = top_k
        self.ocr_dir = getattr(settings, "OCR_DIR", None) or settings.OBJECT_JSON_DIR
        self.ocr_jsonl = os.path.join(self.ocr_dir, "ocr_text.jsonl") if self.ocr_dir else None

    def get_source_name(self) -> str:
        return "ocr"

    def search(
        self,
        query: str,
        video_filter: List[str] = None,
        refined_query: Optional[RefinedQuery] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:

        if not self.ocr_jsonl or not os.path.exists(self.ocr_jsonl):
            # Safe fallback if precomputed OCR file isn't present
            return []

        # Determine target search keywords
        if refined_query and refined_query.analysis.keywords:
            search_keywords = [w.lower() for w in refined_query.analysis.keywords]
        else:
            search_keywords = normalize_text(query).lower().split()

        if not search_keywords:
            return []

        kw_set = set(search_keywords)
        candidates = []

        for item in read_jsonl(self.ocr_jsonl):
            v_id = item.get("video_id", "")
            if video_filter and v_id not in video_filter:
                continue

            ocr_text = normalize_text(item.get("ocr_text", item.get("text", ""))).lower()
            if not ocr_text:
                continue

            words = set(ocr_text.split())
            overlap = len(kw_set & words)
            if overlap > 0:
                score = overlap / (len(kw_set) + 1e-6)
                candidates.append({
                    "video_id": v_id,
                    "frame_id": item.get("frame_id", 0),
                    "score": score,
                    "source": self.get_source_name(),
                    "ocr_text": item.get("ocr_text", item.get("text", ""))
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:self.top_k]