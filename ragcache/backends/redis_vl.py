import time
import uuid
import json
import numpy as np
from typing import Optional

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema
import redis

from .base import AbstractCacheBackend, CacheEntry


SCHEMA = {
    "index": {
        "name": "ragcache",
        "prefix": "ragcache:entry",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "query_text", "type": "text"},
        {"name": "answer", "type": "text"},
        {"name": "chunk_ids_json", "type": "text"},
        {"name": "created_at", "type": "numeric"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": 768,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
    ],
}


class RedisVLBackend(AbstractCacheBackend):
    def __init__(self, redis_url: str = "redis://localhost:6379", dims: int = 768):
        self._redis_url = redis_url
        self._dims = dims
        self._index: Optional[SearchIndex] = None
        self._client: Optional[redis.Redis] = None

    def connect(self) -> None:
        schema = SCHEMA.copy()
        schema["fields"][-1]["attrs"]["dims"] = self._dims

        self._client = redis.from_url(self._redis_url, decode_responses=False)
        index_schema = IndexSchema.from_dict(schema)
        self._index = SearchIndex(index_schema, redis_client=self._client)

        if not self._index.exists():
            self._index.create(overwrite=False)

    def _ensure_connected(self) -> None:
        if self._index is None:
            self.connect()

    def vector_search(self, embedding: np.ndarray, top_k: int = 1) -> list[CacheEntry]:
        self._ensure_connected()

        query = VectorQuery(
            vector=embedding.astype(np.float32).tobytes(),
            vector_field_name="embedding",
            return_fields=["query_text", "answer", "chunk_ids_json", "created_at", "vector_distance"],
            num_results=top_k,
        )

        results = self._index.query(query)
        entries = []
        for r in results:
            similarity = 1.0 - float(r.get("vector_distance", 1.0))
            entries.append(
                CacheEntry(
                    key=r["id"],
                    query_text=r.get("query_text", ""),
                    answer=r.get("answer", ""),
                    embedding=embedding,
                    source_chunk_ids=json.loads(r.get("chunk_ids_json", "[]")),
                    created_at=float(r.get("created_at", 0)),
                    similarity=similarity,
                )
            )
        return entries

    def store(self, entry: CacheEntry) -> None:
        self._ensure_connected()

        key = f"ragcache:entry:{entry.key}"
        self._client.hset(key, mapping={
            "query_text": entry.query_text,
            "answer": entry.answer,
            "chunk_ids_json": json.dumps(entry.source_chunk_ids),
            "created_at": str(entry.created_at),
            "embedding": entry.embedding.astype(np.float32).tobytes(),
        })

    def delete(self, keys: list[str]) -> None:
        self._ensure_connected()
        if keys:
            self._client.delete(*keys)

    def get_keys_for_chunk(self, chunk_id: str) -> list[str]:
        self._ensure_connected()
        reverse_key = f"ragcache:inv:{chunk_id}"
        raw = self._client.smembers(reverse_key)
        return [k.decode() if isinstance(k, bytes) else k for k in raw]

    def add_chunk_reverse_index(self, chunk_id: str, cache_key: str) -> None:
        self._ensure_connected()
        reverse_key = f"ragcache:inv:{chunk_id}"
        self._client.sadd(reverse_key, cache_key)
