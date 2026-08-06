from typing import List, Dict, Any

def reciprocal_rank_fusion(ranked_lists: List[List[Dict[str, Any]]], k: int = 60, top_k: int = 100) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF)
    ranked_lists: list of lists, each list is a list of dicts with at least 'video_id', 'frame_id', and 'score' (or rank)
    """
    scores = {}
    for rank_list in ranked_lists:
        for rank, item in enumerate(rank_list, start=1):
            key = (item["video_id"], item["frame_id"])
            if key not in scores:
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)
    
    # Convert to list of dicts
    merged = []
    for (video_id, frame_id), score in scores.items():
        merged.append({
            "video_id": video_id,
            "frame_id": frame_id,
            "score": score
        })
    
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]