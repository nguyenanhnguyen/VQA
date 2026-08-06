import re
import unicodedata
from typing import List, Optional

def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt: bỏ dấu cách thừa, lowercase, chuẩn hóa unicode."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_similarity(vec1, vec2, method="cosine"):
    """Tính độ tương đồng giữa hai vector."""
    import numpy as np
    if method == "cosine":
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    elif method == "dot":
        return np.dot(vec1, vec2)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def split_sentences(text: str) -> List[str]:
    """Tách câu đơn giản."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]