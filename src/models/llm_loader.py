import requests
import json
from typing import List, Dict, Any
from ..config.settings import settings
from ..utils.logging import logger

def load_llm_model():
    # For local server, we just return the endpoint config
    return {
        "api_base": settings.LLM_API_BASE,
        "api_key": settings.LLM_API_KEY,
        "model": settings.LLM_MODEL_NAME
    }

def call_llm(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    config = load_llm_model()
    api_base = config["api_base"]
    api_key = config["api_key"]
    model = config["model"]
    
    if api_base is None:
        raise ValueError("LLM_API_BASE not set. Cannot call LLM.")
    
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""