from typing import List, Dict, Any
from ..models.llm_loader import call_llm
from ..utils.logging import logger
import json

class Reflection:
    def __init__(self):
        self.prompt_template = """
Bạn là một trợ lý AI đánh giá chất lượng bằng chứng. Hãy xem xét câu hỏi và các bằng chứng được cung cấp, sau đó quyết định xem đã đủ thông tin để trả lời câu hỏi một cách chính xác chưa.

Câu hỏi: {question}

Bằng chứng hiện có:
{evidence}

Hãy trả lời bằng JSON với các trường:
- "sufficient": true/false
- "confidence_score": số từ 0 đến 1 (0: không tin tưởng, 1: hoàn toàn tin tưởng)
- "reason": lý do ngắn gọn
"""
    
    def evaluate(self, question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidence_text = "\n".join([f"- {e.get('description', '')}" for e in evidence])
        prompt = self.prompt_template.format(question=question, evidence=evidence_text)
        messages = [{"role": "user", "content": prompt}]
        
        response = call_llm(messages, temperature=0.0)
        try:
            # Parse JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                result = json.loads(json_str)
            else:
                result = {"sufficient": False, "confidence_score": 0.3, "reason": "Parse error"}
        except Exception as e:
            logger.warning(f"Reflection parse error: {e}")
            result = {"sufficient": False, "confidence_score": 0.3, "reason": "API error"}
        
        return result