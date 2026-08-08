from typing import List, Dict, Any, Optional
from .agents import VisionAgent, OCRAgent, ASRAgent, ObjectAgent, MetadataAgent
from .fusion import reciprocal_rank_fusion
from .reranker import rerank_candidates
from .reflection import Reflection
from ..utils.logging import logger
from ..query_refiner import RefinedQuery, get_query_refiner


def iterative_retrieval(
    question: str,
    context_hints: List[str] = None,
    max_iterations: int = 3,
    refined_query: Optional[RefinedQuery] = None
) -> Dict[str, Any]:
    """
    Iterative retrieval loop with reflection and query decomposition.
    """
    if context_hints is None:
        context_hints = []

    refiner = get_query_refiner()
    if not refined_query:
        refined_query = refiner.refine(question, context_hints=context_hints)

    vision_agent = VisionAgent()
    ocr_agent = OCRAgent()
    asr_agent = ASRAgent()
    object_agent = ObjectAgent()
    metadata_agent = MetadataAgent()
    reflection = Reflection()

    all_candidates = []
    used_hints = []
    video_filter = None  # Could be extended

    current_refined_query = refined_query
    current_question = question

    for iteration in range(max_iterations):
        # Run agents with current refined_query
        vision_results = vision_agent.search(current_question, video_filter=video_filter, refined_query=current_refined_query)
        ocr_results = ocr_agent.search(current_question, video_filter=video_filter, refined_query=current_refined_query)
        asr_results = asr_agent.search(current_question, video_filter=video_filter, refined_query=current_refined_query)
        object_results = object_agent.search(current_question, video_filter=video_filter, refined_query=current_refined_query)
        metadata_results = metadata_agent.search(current_question, video_filter=video_filter, refined_query=current_refined_query)

        # Collect all non-empty results
        all_results = [lst for lst in [vision_results, ocr_results, asr_results, object_results, metadata_results] if lst]

        if not all_results:
            logger.warning("No results from any agent in iteration")
            break

        # Fusion
        fused = reciprocal_rank_fusion(all_results, top_k=100)

        # Rerank
        reranked = rerank_candidates(current_question, fused, top_k=50)

        # Check confidence with Reflection
        evidence = []
        for c in reranked[:10]:
            evidence.append({
                "video_id": c["video_id"],
                "frame_id": c["frame_id"],
                "score": c.get("score", 0),
                "description": f"Video {c['video_id']} frame {c['frame_id']} score {c.get('score',0):.3f}"
            })

        result = reflection.evaluate(current_question, evidence)

        if result.get("sufficient", False):
            return {
                "candidates": reranked,
                "confidence": result.get("confidence_score", 0.5),
                "evidence": evidence,
                "iterations": iteration + 1
            }

        # If not sufficient and has unused hints, decompose query with next hint
        if iteration < len(context_hints):
            next_hint = context_hints[iteration]
            used_hints.append(next_hint)
            current_question = current_question + " " + next_hint
            # Re-refine query including accumulated context hints
            current_refined_query = refiner.refine(question, context_hints=used_hints)
            logger.info(f"Iteration {iteration+1}: refined query with hint '{next_hint[:30]}...' -> sub_queries: {current_refined_query.decomposition.sub_queries}")
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