"""
src/models/clip_loader.py
===========================
Load model embedding dùng cho Tier 0 (single-step retrieval) và
tier2/agents/vision_agent.py.

THIẾT KẾ "Zero-config nhưng Easy-config":
  - Đọc EMBEDDING_MODEL_NAME từ settings.py (không hardcode).
  - Default "ViT-B-32": dùng qua `open_clip`, model公开, nhẹ (~350MB), tải tự
    động lần đầu chạy, KHÔNG cần script/checkpoint đặc biệt -- chạy thử được
    ngay trên máy dev bất kỳ, đúng nguyên tắc zero-config.
  - Khi xác nhận vector trong Milvus thực ra được tạo bằng
    Qwen3-VL-Embedding-8B (model SOTA nêu trong tài liệu nghiên cứu), chỉ cần
    đổi EMBEDDING_MODEL_NAME trong .env, code tự động rẽ nhánh sang loader
    khác -- KHÔNG sửa code caller (tier0.py, vision_agent.py).

   Nhánh Qwen3-VL-Embedding hiện để interface + TODO rõ ràng (xem docstring
   trong hàm _load_qwen3_vl) vì method encode/embed thật cần verify từ README
   đầy đủ sau khi tải checkpoint (xem scripts/download_models.py), tài liệu
   search chỉ có đoạn trích, không phải toàn bộ.
"""

import logging
from typing import List, Optional, Union

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class ClipLoader:
    """Wrapper chung -- caller (tier0.py, vision_agent.py) chỉ cần gọi
    `.encode_text(...)` / `.encode_image(...)`, không cần biết đang chạy
    backend nào (open_clip hay Qwen3-VL-Embedding)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.expected_dim = settings.EMBEDDING_DIM
        self._backend = None
        self._backend_kind = None  # "open_clip" | "qwen3_vl"

    def load(self):
        if self._backend is not None:
            return self

        if self.model_name.lower().startswith("qwen3-vl-embedding") or "qwen3-vl-embedding" in self.model_name.lower():
            self._load_qwen3_vl()
        else:
            self._load_open_clip()
        return self

    # ------------------------------------------------------------------
    # Backend 1: open_clip -- DEFAULT, zero-config, đã kiểm chứng ổn định
    # ------------------------------------------------------------------
    def _load_open_clip(self):
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise ImportError(
                "Cần cài open_clip_torch: thêm 'open-clip-torch' vào requirements.txt "
                "nếu dùng EMBEDDING_MODEL_NAME kiểu CLIP (mặc định)."
            ) from e

        device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        model = model.to(device).eval()

        self._backend = {"model": model, "preprocess": preprocess, "tokenizer": tokenizer, "device": device}
        self._backend_kind = "open_clip"
        logger.info(f"Loaded open_clip model '{self.model_name}' on {device}")

    def _encode_text_open_clip(self, texts: List[str]) -> np.ndarray:
        import torch
        b = self._backend
        with torch.no_grad():
            tokens = b["tokenizer"](texts).to(b["device"])
            features = b["model"].encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def _encode_image_open_clip(self, images) -> np.ndarray:
        """`images`: list các PIL.Image đã load sẵn."""
        import torch
        b = self._backend
        with torch.no_grad():
            batch = torch.stack([b["preprocess"](img) for img in images]).to(b["device"])
            features = b["model"].encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    # ------------------------------------------------------------------
    # Backend 2: Qwen3-VL-Embedding-8B -- production option, CẦN VERIFY THÊM
    # ------------------------------------------------------------------
    def _load_qwen3_vl(self):
        """
        TODO: hoàn thiện sau khi verify method thật từ README đầy đủ của
        Qwen/Qwen3-VL-Embedding-8B (xem scripts/download_models.py để tải
        checkpoint kèm script chính thức `scripts/qwen3_vl_embedding.py`).
        Đã xác nhận qua search: import pattern là
            from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
        nhưng CHƯA xác nhận tên method encode/embed thật + shape output.
        """
        raise NotImplementedError(
            "Backend Qwen3-VL-Embedding chưa hoàn thiện -- cần tải checkpoint "
            "thật và đọc README đầy đủ để xác nhận method API trước khi dùng "
            "production. Xem TODO trong docstring hàm này."
        )

    # ------------------------------------------------------------------
    # Public API -- caller dùng đúng 2 hàm này, không quan tâm backend nào
    # ------------------------------------------------------------------
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        self.load()
        if isinstance(texts, str):
            texts = [texts]
        if self._backend_kind == "open_clip":
            vecs = self._encode_text_open_clip(texts)
        else:
            raise NotImplementedError(f"encode_text chưa hỗ trợ backend '{self._backend_kind}'")
        self._assert_dim(vecs)
        return vecs

    def encode_image(self, images) -> np.ndarray:
        self.load()
        if self._backend_kind == "open_clip":
            vecs = self._encode_image_open_clip(images)
        else:
            raise NotImplementedError(f"encode_image chưa hỗ trợ backend '{self._backend_kind}'")
        self._assert_dim(vecs)
        return vecs

    def _assert_dim(self, vecs: np.ndarray):
        if vecs.shape[-1] != self.expected_dim:
            raise ValueError(
                f"Vector output có dim={vecs.shape[-1]} nhưng EMBEDDING_DIM trong "
                f".env đang là {self.expected_dim}. Kiểm tra lại 2 giá trị này khớp "
                f"đúng model đang dùng (vd ViT-B-32 -> 512, Qwen3-VL-Embedding-8B -> "
                f"kiểm tra README thật, KHÔNG mặc định theo con số đoán)."
            )


_loader_instance: Optional[ClipLoader] = None


def get_clip_loader() -> ClipLoader:
    """Factory lazy-singleton -- không load model ngay lúc import module."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ClipLoader()
        _loader_instance.load()
    return _loader_instance
