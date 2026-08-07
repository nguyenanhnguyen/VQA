from typing import List, Dict, Any, Optional
import json
from ...database.metadata_client import metadata_client
from ...utils.text_utils import normalize_text
from ...utils.file_utils import read_jsonl
from ...config.settings import settings
from .base import BaseAgent
from ...utils.logging import logger
from ...query_refiner import RefinedQuery


class ASRAgent(BaseAgent):
    def __init__(self, top_k: int = 200):
        self.top_k = top_k
        self.transcript_file = settings.TRANSCRIPT_JSONL

    def get_source_name(self) -> str:
        return "asr"

    def search(
        self,
        query: str,
        video_filter: List[str] = None,
        refined_query: Optional[RefinedQuery] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:

        transcripts = []
        for item in read_jsonl(self.transcript_file):
            if video_filter and item.get("video_id") not in video_filter:
                continue
            transcripts.append(item)

        # Sử dụng refined keywords nếu có, giúp loại bỏ hoàn toàn từ nối/từ dừng
        if refined_query and (refined_query.keywords_vi or refined_query.keywords_en):
            query_words = set([w.lower() for w in (refined_query.keywords_vi + refined_query.keywords_en)])
        else:
            query_words = set(normalize_text(query).lower().split())

        if not query_words:
            return []

        scored = []
        for item in transcripts:
            text = normalize_text(item.get("text", ""))
            words = set(text.lower().split())
            overlap = len(query_words & words)
            if overlap > 0:
                score = overlap / (len(query_words) + 1e-6)
                scored.append({
                    "video_id": item["video_id"],
                    "frame_id": 0,
                    "score": score,
                    "source": self.get_source_name(),
                    "asr_text": text
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        candidates = scored[:self.top_k]
        logger.info(f"ASRAgent found {len(candidates)} candidates using keywords: {list(query_words)[:5]}")
        return candidates