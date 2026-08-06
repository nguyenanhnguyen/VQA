from typing import List, Dict, Any, Optional

from .database.milvus_client import get_milvus_client
from .models.clip_loader import get_clip_loader
from .tier2.fusion import reciprocal_rank_fusion
from .utils.logging import logger


def tier0_search(question: str, max_results: int = 100, video_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Tier 0: Simple CLIP + RRF search
    """
    logger.info(f"Tier0: query='{question[:50]}...'")

    embedder = get_clip_loader()
    query_vector = embedder.encode_text(question)[0].tolist()  # encode_text luôn trả batch, lấy vector đầu tiên

    expr = None
    if video_filter:
        video_list = ", ".join([f"'{v}'" for v in video_filter])
        expr = f"video_id in [{video_list}]"

    milvus = get_milvus_client()
    results = milvus.search(query_vector, top_k=max_results, expr=expr)

    candidates = []
    for hit in results:
        frame_id = hit.entity.get("frame_id", 0)
        video_id = hit.entity.get("video_id", "")
        score = hit.score
        candidates.append({
            "video_id": video_id,
            "frame_id": frame_id,
            "score": score,
            "source": "clip",
        })

    ranked = reciprocal_rank_fusion([candidates], top_k=max_results)
    return ranked
