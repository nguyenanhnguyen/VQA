"""
src/query_refiner.py
======================
LLM-based Query Understanding & Query Decomposition Module for VQA.
Replaces legacy heuristic regex query refiner based on query_decomposition_spec.md.

Performs:
1. Canonical English normalization & translation.
2. Question Analysis (intent, modalities, keywords, entities, attributes, actions, temporal_relations).
3. Context/Hint-derived Query Decomposition into semantic retrieval sub-queries.
"""

import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from config.settings import settings

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    original_query: str
    original_language: str = "vi"  # "vi" | "en" | other
    canonical_question: str        # English semantic representation
    intent: str = "general"         # "object_count" | "text_reading" | "speech_content" | "location" | "temporal_event" | "attribute" | "general"
    modalities: List[str] = Field(default_factory=lambda: ["visual"])  # "visual", "ocr", "asr", "od", "metadata"
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    temporal_relations: List[str] = Field(default_factory=list)


class QueryDecomposition(BaseModel):
    needs_decomposition: bool = False
    sub_queries: List[str] = Field(default_factory=list)  # Semantic retrieval sub-queries in English
    relations: List[str] = Field(default_factory=list)    # Causal / temporal relations


class RefinedQuery(BaseModel):
    analysis: QueryAnalysis
    decomposition: QueryDecomposition

    # Backward compatibility properties
    @property
    def original_query(self) -> str:
        return self.analysis.original_query

    @property
    def original_lang(self) -> str:
        return self.analysis.original_language

    @property
    def visual_query_en(self) -> str:
        if self.decomposition.sub_queries:
            return self.decomposition.sub_queries[0]
        return self.analysis.canonical_question

    @property
    def visual_query_vi(self) -> str:
        return self.analysis.original_query

    @property
    def keywords_vi(self) -> List[str]:
        return self.analysis.keywords

    @property
    def keywords_en(self) -> List[str]:
        return self.analysis.keywords

    @property
    def target_attribute(self) -> str:
        if "ocr" in self.analysis.modalities:
            return "text_ocr"
        elif "asr" in self.analysis.modalities:
            return "speech_asr"
        elif "metadata" in self.analysis.modalities:
            return "metadata"
        elif "od" in self.analysis.modalities or self.analysis.intent == "object_count":
            return "object_count"
        elif self.analysis.intent == "action":
            return "action"
        elif "visual" in self.analysis.modalities:
            return "visual"
        return "general"


SYSTEM_PROMPT = """You are an expert Query Analysis and Decomposition agent for a Video Question Answering (VQA) system.
Your job is to analyze user queries (which may be in Vietnamese or English) alongside any context hints, and produce structured information for specialized video retrieval agents.

Return ONLY a valid JSON object matching this exact schema:

{
  "original_language": "vi or en",
  "canonical_question": "Full question translated/normalized into clear English",
  "intent": "One of: object_count, text_reading, speech_content, location, temporal_event, attribute, action, general",
  "modalities": ["Array of required modalities: visual, ocr, asr, od, metadata"],
  "keywords": ["List of atomic English search keywords for text/OCR/ASR/Metadata search"],
  "entities": ["List of physical entities, people, objects mentioned"],
  "attributes": ["List of visual or descriptive attributes like red shirt, tall"],
  "actions": ["List of action verbs like walking, talking, giving bottle"],
  "temporal_relations": ["List of temporal markers like before, after, during"],
  "needs_decomposition": true or false,
  "sub_queries": ["List of independent English semantic retrieval sub-queries representing discrete visual events/anchors"],
  "relations": ["List of relationships between sub_queries, e.g. event_1 before event_2"]
}

RULES:
1. All sub_queries, keywords, canonical_question MUST BE IN ENGLISH.
2. If the user question is simple (e.g. "a red car"), needs_decomposition=false, and sub_queries contains just ["a red car"].
3. If the user question is complex or relational (e.g., "The man in black shirt gives water to woman then enters store"), needs_decomposition=true, and sub_queries should break it down into semantic retrieval units:
   ["man wearing black shirt", "man giving water bottle to woman", "man entering store"]
4. Use context_hints if provided to resolve pronouns (e.g. "he" -> "man wearing black shirt").
5. Keep sub_queries informative enough to serve as standalone CLIP visual prompts (do NOT break into tiny meaningless single words).
6. Do NOT include markdown codeblocks or extra conversational text outside the JSON.
"""


class LLMQueryRefiner:
    def __init__(self):
        self.provider = getattr(settings, "QUERY_ANALYZER_PROVIDER", "gemini")
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
        self.hf_token = os.environ.get("HF_TOKEN") or getattr(settings, "HF_TOKEN", None)

    def _call_gemini(self, prompt: str) -> Optional[str]:
        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            model_name = getattr(settings, "QUERY_ANALYZER_MODEL", "gemini-2.5-flash-lite")
            # Try interaction API or chat/completion API
            response = client.interactions.create(
                model=model_name,
                input=f"{SYSTEM_PROMPT}\n\n{prompt}"
            )
            return response.output_text
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}")
            return None

    def _call_hf(self, prompt: str) -> Optional[str]:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.hf_token
            )
            model_name = getattr(settings, "HF_QUERY_ANALYZER_MODEL", "Qwen/Qwen2.5-7B-Instruct:together")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"HF OpenAI API call failed: {e}")
            return None

    def _call_fallback_llm(self, prompt: str) -> Optional[str]:
        """Call standard local/OpenAI LLM via project's model loader."""
        try:
            from .models.llm_loader import call_llm
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            return call_llm(messages, temperature=0.1)
        except Exception as e:
            logger.warning(f"Fallback LLM call failed: {e}")
            return None

    def parse_llm_output(self, raw_query: str, raw_output: str) -> RefinedQuery:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in output: {raw_output[:100]}...")

        data = json.loads(match.group(0))

        analysis = QueryAnalysis(
            original_query=raw_query,
            original_language=data.get("original_language", "vi"),
            canonical_question=data.get("canonical_question", raw_query),
            intent=data.get("intent", "general"),
            modalities=data.get("modalities", ["visual"]),
            keywords=data.get("keywords", []),
            entities=data.get("entities", []),
            attributes=data.get("attributes", []),
            actions=data.get("actions", []),
            temporal_relations=data.get("temporal_relations", []),
        )

        needs_decomp = data.get("needs_decomposition", False)
        sub_queries = data.get("sub_queries", [])
        if not sub_queries:
            sub_queries = [analysis.canonical_question]

        decomposition = QueryDecomposition(
            needs_decomposition=needs_decomp,
            sub_queries=sub_queries,
            relations=data.get("relations", [])
        )

        return RefinedQuery(analysis=analysis, decomposition=decomposition)

    def _fallback_refined_query(self, query: str) -> RefinedQuery:
        """Safe fallback if all LLM options fail."""
        words = [w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) > 2]
        analysis = QueryAnalysis(
            original_query=query,
            original_language="vi" if any(c in "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ" for c in query) else "en",
            canonical_question=query,
            intent="general",
            modalities=["visual"],
            keywords=words,
            entities=words,
            attributes=[],
            actions=[],
            temporal_relations=[],
        )
        decomposition = QueryDecomposition(
            needs_decomposition=False,
            sub_queries=[query],
            relations=[]
        )
        return RefinedQuery(analysis=analysis, decomposition=decomposition)

    def refine(self, query: str, context_hints: Optional[List[str]] = None) -> RefinedQuery:
        user_prompt = f"User Question: \"{query}\""
        if context_hints:
            user_prompt += f"\nContext Hints / Prior Conversation: {json.dumps(context_hints, ensure_ascii=False)}"

        raw_output = None

        # 1. Try primary provider (Gemini or configured)
        if self.gemini_key:
            raw_output = self._call_gemini(user_prompt)

        # 2. Try HF Qwen2.5-7B if primary failed or key not available
        if not raw_output and self.hf_token:
            raw_output = self._call_hf(user_prompt)

        # 3. Try standard local/OpenAI LLM
        if not raw_output:
            raw_output = self._call_fallback_llm(user_prompt)

        if raw_output:
            try:
                refined = self.parse_llm_output(query, raw_output)
                logger.info(f"QueryRefiner success. Canonical: '{refined.analysis.canonical_question}' | Sub-queries: {refined.decomposition.sub_queries}")
                return refined
            except Exception as e:
                logger.warning(f"Failed to parse LLM query refiner output: {e}")

        logger.warning(f"Using fallback query refinement for: '{query}'")
        return self._fallback_refined_query(query)


_refiner_instance: Optional[LLMQueryRefiner] = None


def get_query_refiner() -> LLMQueryRefiner:
    global _refiner_instance
    if _refiner_instance is None:
        _refiner_instance = LLMQueryRefiner()
    return _refiner_instance
