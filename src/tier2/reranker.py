from typing import List, Dict, Any
from ..models.reranker_loader import rerank_candidates
from ..utils.logging import logger

def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Rerank candidates using Cross-Encoder
    """
    if not candidates:
        return []
    
    # Prepare pairs
    # For each candidate, we need a text description
    # We can use metadata or OCR/ASR text if available, else just video_id/frame_id
    pairs = []
    for c in candidates:
        # Tạo description từ các nguồn có sẵn
        desc = f"Video {c['video_id']} frame {c['frame_id']}"
        pairs.append((c['video_id'], c['frame_id'], desc))
    
    # Rerank
    scores = rerank_candidates(query, pairs)
    
    # Update scores
    for i, c in enumerate(candidates):
        c["rerank_score"] = scores[i]
    
    # Sort by rerank score
    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]