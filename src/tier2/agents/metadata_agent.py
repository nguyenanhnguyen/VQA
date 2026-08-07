from typing import List, Dict, Any, Optional
from ...database.metadata_client import metadata_client
from ...utils.text_utils import normalize_text
from .base import BaseAgent
from ...utils.logging import logger
from ...query_refiner import RefinedQuery


class MetadataAgent(BaseAgent):
    def __init__(self, top_k: int = 100):
        self.top_k = top_k

    def get_source_name(self) -> str:
        return "metadata"

    def search(
        self,
        query: str,
        video_filter: List[str] = None,
        refined_query: Optional[RefinedQuery] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:

        candidates = []
        if video_filter is None:
            logger.warning("MetadataAgent requires video_filter")
            return candidates

        if refined_query and (refined_query.keywords_vi or refined_query.keywords_en):
            query_words = set([w.lower() for w in (refined_query.keywords_vi + refined_query.keywords_en)])
        else:
            query_words = set(normalize_text(query).lower().split())

        if not query_words:
            return []

        for video_id in video_filter:
            meta = metadata_client.get_metadata(video_id)
            if meta is None:
                continue

            text = (meta.get("title", "") + " " + meta.get("description", "") + " " + " ".join(meta.get("keywords", [])))
            text = normalize_text(text.lower())
            words = set(text.split())
            overlap = len(query_words & words)
            if overlap > 0:
                score = overlap / (len(query_words) + 1e-6)
                candidates.append({
                    "video_id": video_id,
                    "frame_id": 0,
                    "score": score,
                    "source": self.get_source_name(),
                    "description": meta.get("description", meta.get("title", "")),
                    "metadata": meta
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:self.top_k]