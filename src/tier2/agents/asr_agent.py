from typing import List, Dict, Any
import json
from ...database.metadata_client import metadata_client
from ...utils.text_utils import normalize_text
from ...utils.file_utils import read_jsonl
from ...config.settings import settings
from .base import BaseAgent
from ...utils.logging import logger

class ASRAgent(BaseAgent):
    def __init__(self, top_k: int = 200):
        self.top_k = top_k
        self.transcript_file = settings.TRANSCRIPT_JSONL
    
    def get_source_name(self) -> str:
        return "asr"
    
    def search(self, query: str, video_filter: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        # Load transcripts from file (có thể cache)
        transcripts = []
        for item in read_jsonl(self.transcript_file):
            if video_filter and item.get("video_id") not in video_filter:
                continue
            transcripts.append(item)
        
        # Simple keyword matching
        query_words = set(normalize_text(query).lower().split())
        scored = []
        for item in transcripts:
            text = normalize_text(item.get("text", ""))
            words = set(text.lower().split())
            overlap = len(query_words & words)
            if overlap > 0:
                # Heuristic score: overlap count
                score = overlap / (len(query_words) + 1e-6)
                scored.append({
                    "video_id": item["video_id"],
                    "frame_id": 0,  # ASR không gắn frame cụ thể, có thể dùng shot_id
                    "score": score,
                    "source": self.get_source_name(),
                    "text": text
                })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        candidates = scored[:self.top_k]
        logger.info(f"ASRAgent found {len(candidates)} candidates")
        return candidates