from typing import Dict, Any
import time

class Cache:
    def __init__(self, ttl_seconds: int = 3600):
        self.data: Dict[str, Any] = {}
        self.ttl = ttl_seconds
        self.timestamps: Dict[str, float] = {}
    
    def get(self, key: str):
        if key not in self.data:
            return None
        if time.time() - self.timestamps.get(key, 0) > self.ttl:
            del self.data[key]
            del self.timestamps[key]
            return None
        return self.data[key]
    
    def set(self, key: str, value: Any):
        self.data[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        self.data.clear()
        self.timestamps.clear()

cache = Cache()