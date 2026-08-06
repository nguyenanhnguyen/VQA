"""
src/database/milvus_client.py
===============================
Client kết nối Milvus (Vector DB), đọc toàn bộ config từ settings.py.

QUAN TRỌNG: KHÔNG khởi tạo kết nối ngay lúc import module (khác bản gốc bạn
dán). Lý do: nếu module-level tự connect(), bất kỳ file nào import
milvus_client.py (kể cả test không liên quan tới Milvus) đều bị bắt buộc kết
nối thật, làm test suite fail vô lý khi Milvus chưa chạy. Dùng factory
get_milvus_client() để lazy-init: chỉ kết nối khi thực sự cần dùng lần đầu.
"""

import logging
from typing import List, Optional

from pymilvus import connections, Collection, utility

from config.settings import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION
        self.vector_field = settings.MILVUS_VECTOR_FIELD
        self.id_field = settings.MILVUS_ID_FIELD
        self.metadata_fields = settings.MILVUS_METADATA_FIELDS.split(",")
        self.collection: Optional[Collection] = None
        self._connected = False

    def connect(self):
        """Kết nối + load collection. Gọi tường minh (hoặc tự động qua search())
        thay vì chạy ngầm lúc import module."""
        if self._connected:
            return

        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise RuntimeError(
                f"Không kết nối được Milvus tại {self.host}:{self.port}. "
                f"Kiểm tra MILVUS_HOST/MILVUS_PORT trong .env và Milvus server có đang chạy không."
            ) from e

        if not utility.has_collection(self.collection_name):
            raise RuntimeError(
                f"Collection '{self.collection_name}' không tồn tại trên Milvus. "
                f"Xác nhận lại MILVUS_COLLECTION trong .env khớp với collection DLong đã tạo."
            )

        self.collection = Collection(self.collection_name)
        self.collection.load()
        self._connected = True
        logger.info(f"Loaded Milvus collection: {self.collection_name}")

    def search(self, query_vector: List[float], top_k: int = 100, expr: str = None,
               output_fields: Optional[List[str]] = None):
        self.connect()  # lazy: tự connect ở lần gọi đầu tiên nếu chưa connect

        if output_fields is None:
            output_fields = [self.id_field] + self.metadata_fields

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field=self.vector_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )
        return results[0]  # list of hits


_client_instance: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """Factory — dùng hàm này thay vì tạo `MilvusClient()` trực tiếp ở nơi khác,
    đảm bảo chỉ có đúng 1 instance/kết nối dùng chung toàn app."""
    global _client_instance
    if _client_instance is None:
        _client_instance = MilvusClient()
    return _client_instance
