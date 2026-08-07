"""
src/query_refiner.py
======================
Bilingual Query Refinement & Keyword Extraction Module for VQA.
Cleans raw conversational queries into simplified visual prompts for CLIP embeddings
(both Vietnamese and English) and extracts key search terms for ASR, OCR, Object,
and Metadata agents.

Supports both:
  1. Fast Regex Heuristic method (< 2ms, zero-config default).
  2. Small LLM method (optional, calls LLM to produce JSON breakdown for complex queries).
"""

import re
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from config.settings import settings
from .models.llm_loader import call_llm

logger = logging.getLogger(__name__)


class RefinedQuery(BaseModel):
    original_query: str
    original_lang: str = "vi"  # "vi" or "en"
    visual_query_vi: str
    visual_query_en: str
    keywords_vi: List[str] = Field(default_factory=list)
    keywords_en: List[str] = Field(default_factory=list)
    target_attribute: str = "general"  # "text_ocr" | "speech_asr" | "metadata" | "object_count" | "action" | "visual" | "general"


# Stop phrases to strip out conversational filler
VI_STOP_PHRASES = [
    r"cho tôi biết", r"hãy cho biết", r"cho biết", r"bạn có biết", r"hãy tìm",
    r"hãy cho tôi biết", r"trong video này", r"trong clip này", r"ở phút thứ \d+",
    r"ở giây thứ \d+", r"khoảnh khắc", r"là gì\??", r"như thế nào\??", r"đang làm gì\??"
]

EN_STOP_PHRASES = [
    r"please tell me", r"can you tell me", r"do you know", r"what is", r"what are",
    r"in this video", r"in the clip", r"at minute \d+", r"at second \d+",
    r"what is happening", r"find the", r"show me"
]


class BilingualQueryRefiner:
    def __init__(self):
        pass

    def detect_language(self, text: str) -> str:
        """Detect whether text is primarily Vietnamese or English."""
        vi_chars = set("àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ")
        if any(c in vi_chars for c in text):
            return "vi"
        return "en"

    def refine_heuristic(self, query: str) -> RefinedQuery:
        """Fast Regex Heuristic Method (< 2ms, no LLM required)."""
        clean_query = query.strip()
        lang = self.detect_language(clean_query)

        # 1. Detect target attribute intent
        lower = clean_query.lower()
        target_attr = "general"
        if any(w in lower for w in ["chữ", "biển báo", "dòng chữ", "viết gì", "text", "sign", "written", "reads"]):
            target_attr = "text_ocr"
        elif any(w in lower for w in ["nói", "lời thoại", "phát biểu", "âm thanh", "say", "said", "speech", "talk", "transcript"]):
            target_attr = "speech_asr"
        elif any(w in lower for w in ["tiêu đề", "kênh", "tác giả", "ngày đăng", "title", "channel", "author", "metadata"]):
            target_attr = "metadata"
        elif any(w in lower for w in ["bao nhiêu", "mấy", "đếm", "số lượng", "how many", "count"]):
            target_attr = "object_count"
        elif any(w in lower for w in ["đang làm gì", "hành động", "chạy", "nhảy", "đi bộ", "doing", "action", "running", "walking"]):
            target_attr = "action"
        elif any(w in lower for w in ["màu", "áo", "quần", "xe", "vật", "color", "shirt", "wearing", "holding", "car"]):
            target_attr = "visual"

        # 2. Strip conversational filler
        vis_vi = lower
        for pattern in VI_STOP_PHRASES:
            vis_vi = re.sub(pattern, "", vis_vi, flags=re.IGNORECASE)

        vis_en = lower
        for pattern in EN_STOP_PHRASES:
            vis_en = re.sub(pattern, "", vis_en, flags=re.IGNORECASE)

        vis_vi = re.sub(r"\s+", " ", vis_vi).strip(" ?,.")
        vis_en = re.sub(r"\s+", " ", vis_en).strip(" ?,.")

        if not vis_vi:
            vis_vi = clean_query
        if not vis_en:
            vis_en = clean_query

        # 3. Extract keywords
        words = re.findall(r"\b\w+\b", lower)
        stop_words = {"là", "gì", "ở", "trong", "có", "nào", "này", "đó", "cho", "tôi", "biết", "của", "và", "với", "phút", "giây", "what", "is", "in", "the", "a", "an", "on", "at", "of", "and", "or", "to", "this"}
        kw_vi = [w for w in words if w not in stop_words and len(w) > 1]
        kw_en = kw_vi

        return RefinedQuery(
            original_query=query,
            original_lang=lang,
            visual_query_vi=vis_vi,
            visual_query_en=vis_en if lang == "en" else vis_vi,
            keywords_vi=kw_vi,
            keywords_en=kw_en,
            target_attribute=target_attr
        )

    def refine_llm(self, query: str) -> Optional[RefinedQuery]:
        """Optional Small LLM Method (Structured JSON extraction)."""
        prompt = (
            f"Analyze the following video QA query and extract structured components in JSON format:\n"
            f"Query: \"{query}\"\n\n"
            f"Output JSON format with keys:\n"
            f"- original_lang: 'vi' or 'en'\n"
            f"- visual_query_vi: simplified Vietnamese visual prompt (strip filler)\n"
            f"- visual_query_en: simplified English visual prompt\n"
            f"- keywords_vi: list of key Vietnamese search terms\n"
            f"- keywords_en: list of key English search terms\n"
            f"- target_attribute: 'text_ocr', 'speech_asr', 'metadata', 'object_count', 'action', 'visual', or 'general'\n"
            f"JSON Output:"
        )

        messages = [
            {"role": "system", "content": "You are a specialized query parsing assistant for Video QA. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            res_str = call_llm(messages, temperature=0.0)
            # Find JSON block
            match = re.search(r"\{.*\}", res_str, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return RefinedQuery(
                    original_query=query,
                    original_lang=data.get("original_lang", "vi"),
                    visual_query_vi=data.get("visual_query_vi", query),
                    visual_query_en=data.get("visual_query_en", query),
                    keywords_vi=data.get("keywords_vi", []),
                    keywords_en=data.get("keywords_en", []),
                    target_attribute=data.get("target_attribute", "general")
                )
        except Exception as e:
            logger.warning(f"QueryRefiner LLM extraction failed: {e}. Falling back to heuristic.")
        return None

    def refine(self, query: str) -> RefinedQuery:
        """
        Main entrypoint.
        Tries LLM refinement if enabled in settings & LLM_API_BASE is available,
        otherwise uses fast Heuristic refinement.
        """
        use_llm = getattr(settings, "ENABLE_LLM_QUERY_REFINEMENT", False) and settings.LLM_API_BASE
        if use_llm:
            res = self.refine_llm(query)
            if res:
                return res

        return self.refine_heuristic(query)


_refiner_instance: Optional[BilingualQueryRefiner] = None


def get_query_refiner() -> BilingualQueryRefiner:
    global _refiner_instance
    if _refiner_instance is None:
        _refiner_instance = BilingualQueryRefiner()
    return _refiner_instance
