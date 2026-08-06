from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseAgent(ABC):
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        pass