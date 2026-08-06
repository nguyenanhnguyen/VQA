import time
from collections import defaultdict
from typing import Dict, Any

class MetricsCollector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.counts = defaultdict(int)
        self.timings = defaultdict(float)
        self.results = []
    
    def record(self, key: str, value: Any = None):
        if key.startswith("time_"):
            self.timings[key] += value
        else:
            self.counts[key] += 1
            if value is not None:
                self.results.append({key: value})
    
    def get_report(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        report = {
            "elapsed_seconds": elapsed,
            "counts": dict(self.counts),
            "timings": dict(self.timings)
        }
        return report

metrics = MetricsCollector()