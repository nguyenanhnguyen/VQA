import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import List, Optional
from ..config.settings import settings
from ..utils.logging import logger

_model = None
_processor = None

def load_vlm_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    
    model_name = settings.VLM_MODEL_NAME
    logger.info(f"Loading VLM model: {model_name}")
    
    device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    _processor = AutoProcessor.from_pretrained(model_name)
    _model.eval()
    logger.info(f"VLM loaded on {device}")
    return _model, _processor

def generate_answer(question: str, images: List[Image.Image], max_new_tokens: int = 100) -> str:
    model, processor = load_vlm_model()
    if not images:
        return "Không có đủ hình ảnh để trả lời."
    
    # Ghép các ảnh thành một prompt
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img} for img in images
        ] + [{"type": "text", "text": question}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt, return_tensors="pt")
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    # Lấy phần sau prompt
    answer = answer.split("assistant\n")[-1].strip()
    return answer