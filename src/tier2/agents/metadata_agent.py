from typing import List, Dict, Any
from ...database.metadata_client import metadata_client
from ...utils.text_utils import normalize_text
from .base import BaseAgent
from ...utils.logging import logger

class MetadataAgent(BaseAgent):
    def __init__(self, top_k: int = 100):
        self.top_k = top_k
    
    def get_source_name(self) -> str:
        return "metadata"
    
    def search(self, query: str, video_filter: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        # Search in metadata (title, description, keywords)
        # Need list of video_ids to search
        # Tạm thời dùng video_filter, nếu không có thì không tìm được
        candidates = []
        if video_filter is None:
            logger.warning("MetadataAgent requires video_filter")
            return candidates
        
        query_words = set(normalize_text(query).lower().split())
        for video_id in video_filter:
            meta = metadata_client.get_metadata(video_id)
            if meta is None:
                continue
            # Combine fields
            text = (meta.get("title", "") + " " + meta.get("description", "") + " " + " ".join(meta.get("keywords", [])))
            text = normalize_text(text.lower())
            words = set(text.split())
            overlap = len(query_words & words)
            if overlap > 0:
                score = overlap / (len(query_words) + 1e-6)
                candidates.append({
                    "video_id": video_id,
                    "frame_id": 0,  # metadata áp dụng cho cả video
                    "score": score,
                    "source": self.get_source_name(),
                    "metadata": meta
                })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:self.top_k]