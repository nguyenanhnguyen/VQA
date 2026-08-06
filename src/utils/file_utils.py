import os
import json
from typing import List, Dict, Any, Iterator
from PIL import Image
import cv2

def read_jsonl(filepath: str) -> Iterator[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(filepath: str, data: List[Dict[str, Any]]):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_frame_as_cv2(path: str):
    return cv2.imread(path)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)