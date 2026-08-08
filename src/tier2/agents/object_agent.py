import os
from typing import List, Dict, Any, Optional
from ...utils.text_utils import normalize_text
from ...utils.file_utils import read_jsonl
from ...config.settings import settings
from .base import BaseAgent
from ...utils.logging import logger
from ...query_refiner import RefinedQuery


class ObjectAgent(BaseAgent):
    def __init__(self, top_k: int = 200):
        self.top_k = top_k
        self.object_dir = settings.OBJECT_JSON_DIR
        self.object_jsonl = os.path.join(self.object_dir, "detected_objects.jsonl") if self.object_dir else None

    def get_source_name(self) -> str:
        return "object"

    def search(
        self,
        query: str,
        video_filter: List[str] = None,
        refined_query: Optional[RefinedQuery] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:

        if not self.object_jsonl or not os.path.exists(self.object_jsonl):
            return []

        # Extract target object labels from entities and keywords
        target_labels = set()
        if refined_query:
            target_labels.update([e.lower() for e in refined_query.analysis.entities])
            target_labels.update([k.lower() for k in refined_query.analysis.keywords])
        else:
            target_labels.update(normalize_text(query).lower().split())

        if not target_labels:
            return []

        candidates = []
        for item in read_jsonl(self.object_jsonl):
            v_id = item.get("video_id", "")
            if video_filter and v_id not in video_filter:
                continue

            detected = [obj.get("label", "").lower() for obj in item.get("objects", [])]
            detected_set = set(detected)

            overlap = len(target_labels & detected_set)
            if overlap > 0:
                score = overlap / (len(target_labels) + 1e-6)
                candidates.append({
                    "video_id": v_id,
                    "frame_id": item.get("frame_id", 0),
                    "score": score,
                    "source": self.get_source_name(),
                    "detected_objects": detected
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:self.top_k]