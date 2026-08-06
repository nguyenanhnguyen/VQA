from typing import List, Dict, Any
from .agents import VisionAgent, OCRAgent, ASRAgent, ObjectAgent, MetadataAgent
from .fusion import reciprocal_rank_fusion
from .reranker import rerank_candidates
from .reflection import Reflection
from ..utils.logging import logger

def iterative_retrieval(question: str, context_hints: List[str], max_iterations: int = 3) -> Dict[str, Any]:
    """
    Iterative retrieval loop with reflection
    """
    vision_agent = VisionAgent()
    ocr_agent = OCRAgent()
    asr_agent = ASRAgent()
    object_agent = ObjectAgent()
    metadata_agent = MetadataAgent()
    reflection = Reflection()
    
    all_candidates = []
    used_hints = []
    video_filter = None  # Có thể mở rộng
    
    for iteration in range(max_iterations):
        # Run agents
        vision_results = vision_agent.search(question, video_filter=video_filter)
        ocr_results = ocr_agent.search(question, video_filter=video_filter)
        asr_results = asr_agent.search(question, video_filter=video_filter)
        object_results = object_agent.search(question, video_filter=video_filter)
        metadata_results = metadata_agent.search(question, video_filter=video_filter)
        
        # Collect all results
        all_results = [vision_results, ocr_results, asr_results, object_results, metadata_results]
        # Remove empty lists
        all_results = [lst for lst in all_results if lst]
        
        if not all_results:
            logger.warning("No results from any agent")
            break
        
        # Fusion
        fused = reciprocal_rank_fusion(all_results, top_k=100)
        
        # Rerank
        reranked = rerank_candidates(question, fused, top_k=50)
        
        # Check confidence
        evidence = []
        for c in reranked[:10]:
            evidence.append({
                "video_id": c["video_id"],
                "frame_id": c["frame_id"],
                "score": c.get("score", 0),
                "description": f"Video {c['video_id']} frame {c['frame_id']} score {c.get('score',0):.3f}"
            })
        
        result = reflection.evaluate(question, evidence)
        
        if result.get("sufficient", False):
            return {
                "candidates": reranked,
                "confidence": result.get("confidence_score", 0.5),
                "evidence": evidence,
                "iterations": iteration + 1
            }
        
        # If not sufficient and has hints, use next hint
        if iteration < len(context_hints):
            next_hint = context_hints[iteration]
            used_hints.append(next_hint)
            # Modify question or filter based on hint
            question = question + " " + next_hint
            logger.info(f"Iteration {iteration+1}: added hint '{next_hint[:30]}...'")
        else:
            break
    
    # Final fallback
    return {
        "candidates": reranked if 'reranked' in locals() else [],
        "confidence": 0.2,
        "evidence": evidence if 'evidence' in locals() else [],
        "iterations": iteration + 1,
        "warning": "Max iterations reached without sufficient confidence"
    }