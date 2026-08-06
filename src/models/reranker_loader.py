from sentence_transformers import CrossEncoder
from typing import List, Tuple
from ..config.settings import settings
from ..utils.logging import logger

_model = None

def load_reranker_model():
    global _model
    if _model is not None:
        return _model
    model_name = settings.RERANKER_MODEL_NAME
    logger.info(f"Loading reranker model: {model_name}")
    _model = CrossEncoder(model_name, device="cuda" if settings.USE_GPU else "cpu")
    return _model

def rerank_candidates(query: str, candidates: List[Tuple[str, str]]) -> List[float]:
    """
    candidates: list of (video_id, frame_id, text_description)
    Return list of scores (higher = better)
    """
    model = load_reranker_model()
    pairs = [(query, desc) for _, _, desc in candidates]
    scores = model.predict(pairs)
    return scores.tolist()