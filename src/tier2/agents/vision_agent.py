from typing import List, Dict, Any
from ...database.milvus_client import milvus_client
from ...models.clip_loader import get_clip_embedding
from .base import BaseAgent
from ...utils.logging import logger

class VisionAgent(BaseAgent):
    def __init__(self, top_k: int = 500):
        self.top_k = top_k
    
    def get_source_name(self) -> str:
        return "vision"
    
    def search(self, query: str, video_filter: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        embedding = get_clip_embedding(query)
        if embedding is None:
            logger.warning("VisionAgent: CLIP embedding failed")
            return []
        
        expr = None
        if video_filter:
            video_list = ", ".join([f"'{v}'" for v in video_filter])
            expr = f"video_id in [{video_list}]"
        
        results = milvus_client.search(
            embedding.tolist(),
            top_k=self.top_k,
            expr=expr,
            output_fields=["video_id", "frame_id"]
        )
        
        candidates = []
        for hit in results:
            candidates.append({
                "video_id": hit.entity.get("video_id", ""),
                "frame_id": hit.entity.get("frame_id", 0),
                "score": hit.score,
                "source": self.get_source_name()
            })
        logger.info(f"VisionAgent found {len(candidates)} candidates")
        return candidates