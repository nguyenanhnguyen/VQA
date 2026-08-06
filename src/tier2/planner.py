import json
from typing import Dict, List, Any
from ..models.llm_loader import call_llm
from ..config.settings import settings
from ..utils.logging import logger
import yaml
import os

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "../../config/prompts.yaml")

def load_prompts():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

prompts = load_prompts()

class Planner:
    def __init__(self):
        self.prompt_template = prompts.get("planner_prompt", "")
    
    def plan(self, question: str, context_hints: List[str] = None) -> Dict[str, Any]:
        context = " ".join(context_hints) if context_hints else ""
        prompt = self.prompt_template.format(question=question, context=context)
        
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages, temperature=0.0)
        
        # Parse response (expect JSON)
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                plan = json.loads(json_str)
            else:
                plan = {"entities": [], "attributes": [], "relations": [], "sources": []}
        except:
            plan = {"entities": [], "attributes": [], "relations": [], "sources": []}
        
        logger.info(f"Planner output: {plan}")
        return plan