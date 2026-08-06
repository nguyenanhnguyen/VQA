import os
import json
from typing import Dict, Any, Optional
from ..config.settings import settings
from ..utils.logging import logger

class MetadataClient:
    def __init__(self):
        self.metadata_dir = settings.METADATA_DIR
        self.cache = {}
    
    def get_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        if video_id in self.cache:
            return self.cache[video_id]
        
        metadata_file = os.path.join(self.metadata_dir, f"{video_id}.json")
        if not os.path.exists(metadata_file):
            logger.warning(f"Metadata not found for video: {video_id}")
            return None
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cache[video_id] = data
        return data
    
    def get_field(self, video_id: str, field: str, default=None):
        meta = self.get_metadata(video_id)
        if meta is None:
            return default
        return meta.get(field, default)

metadata_client = MetadataClient()