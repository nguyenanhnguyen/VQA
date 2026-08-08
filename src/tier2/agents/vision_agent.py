from typing import List, Dict, Any, Optional
from ...database.milvus_client import get_milvus_client
from ...models.clip_loader import get_clip_loader
from .base import BaseAgent
from ...utils.logging import logger
from ...query_refiner import RefinedQuery
from ..fusion import reciprocal_rank_fusion


class VisionAgent(BaseAgent):
    def __init__(self, top_k: int = 500):
        self.top_k = top_k

    def get_source_name(self) -> str:
        return "vision"

    def search(
        self,
        query: str,
        video_filter: List[str] = None,
        refined_query: Optional[RefinedQuery] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:

        embedder = get_clip_loader()
        milvus = get_milvus_client()

        # Build list of retrieval sub-queries to search
        sub_queries = []
        if refined_query and refined_query.decomposition and refined_query.decomposition.sub_queries:
            sub_queries = refined_query.decomposition.sub_queries
        elif refined_query and refined_query.analysis.canonical_question:
            sub_queries = [refined_query.analysis.canonical_question]
        else:
            sub_queries = [query]

        expr = None
        if video_filter:
            video_list = ", ".join([f"'{v}'" for v in video_filter])
            expr = f"video_id in [{video_list}]"

        all_subquery_results = []

        for sq in sub_queries:
            vectors = embedder.encode_text(sq)
            if vectors is None or len(vectors) == 0:
                continue
            query_vector = vectors[0].tolist()

            hits = milvus.search(
                query_vector,
                top_k=self.top_k,
                expr=expr
            )

            sq_candidates = []
            for hit in hits:
                sq_candidates.append({
                    "video_id": hit.entity.get("video_id", ""),
                    "frame_id": hit.entity.get("frame_id", 0),
                    "score": hit.score,
                    "source": self.get_source_name(),
                    "sub_query": sq
                })
            all_subquery_results.append(sq_candidates)

        if not all_subquery_results:
            logger.warning("VisionAgent: No valid CLIP embeddings/results produced.")
            return []

        # Fuse sub-query candidates if multiple sub-queries exist
        if len(all_subquery_results) == 1:
            candidates = all_subquery_results[0]
        else:
            candidates = reciprocal_rank_fusion(all_subquery_results, top_k=self.top_k)

        logger.info(f"VisionAgent retrieved {len(candidates)} candidates from {len(sub_queries)} sub-queries: {sub_queries[:3]}")
        return candidates