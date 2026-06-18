import time
import uuid
import functools
from dataclasses import dataclass
from typing import Optional, Callable, Any

import numpy as np

from .backends.base import AbstractCacheBackend, CacheEntry
from .backends.redis_vl import RedisVLBackend
from .embedders.base import AbstractEmbedder
from .embedders.sentence_transformers import SentenceTransformerEmbedder


@dataclass
class CacheResult:
    answer: str
    source_chunk_ids: list[str]
    similarity: float
    cache_key: str
    latency_ms: float


class SemanticCache:
    """
    Drop-in semantic cache for RAG pipelines.

    Usage:
        cache = SemanticCache(redis_url="redis://localhost:6379")

        result = cache.lookup("what is your refund policy?")
        if result:
            return result.answer  # ~5ms, no LLM call

        answer, chunk_ids = my_rag_pipeline(query)
        cache.store(query, answer, chunk_ids)
        return answer
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        similarity_threshold: float = 0.90,
        ttl: int = 86400,
        embedder: Optional[AbstractEmbedder] = None,
        backend: Optional[AbstractCacheBackend] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl

        self._embedder = embedder or SentenceTransformerEmbedder()

        if backend is None:
            # connect() is deferred until first use so import doesn't fail without Redis
            self._backend = RedisVLBackend(redis_url=redis_url, dims=self._embedder.dimensions)
        else:
            self._backend = backend

        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            self._backend.connect()
            self._connected = True

    def lookup(self, query: str) -> Optional[CacheResult]:
        """Check the cache. Returns CacheResult on hit, None on miss."""
        self._ensure_connected()
        t0 = time.perf_counter()

        embedding = self._embedder.embed(query)
        results = self._backend.vector_search(embedding, top_k=1)

        latency_ms = (time.perf_counter() - t0) * 1000

        if not results:
            return None

        top = results[0]
        if top.similarity >= self.similarity_threshold:
            return CacheResult(
                answer=top.answer,
                source_chunk_ids=top.source_chunk_ids,
                similarity=top.similarity,
                cache_key=top.key,
                latency_ms=latency_ms,
            )
        return None

    def store(self, query: str, answer: str, source_chunk_ids: list[str]) -> str:
        """Store a query→answer pair. Returns the cache key."""
        self._ensure_connected()

        embedding = self._embedder.embed(query)
        key = str(uuid.uuid4())

        entry = CacheEntry(
            key=key,
            query_text=query,
            answer=answer,
            embedding=embedding,
            source_chunk_ids=source_chunk_ids,
            created_at=time.time(),
        )
        self._backend.store(entry)

        for chunk_id in source_chunk_ids:
            self._backend.add_chunk_reverse_index(chunk_id, f"ragcache:entry:{key}")

        return key

    def invalidate(self, chunk_ids: list[str]) -> int:
        """
        Delete all cache entries that were generated from the given chunk IDs.
        Returns count of entries invalidated.
        """
        self._ensure_connected()
        keys_to_delete: set[str] = set()

        for chunk_id in chunk_ids:
            keys = self._backend.get_keys_for_chunk(chunk_id)
            keys_to_delete.update(keys)

        if keys_to_delete:
            self._backend.delete(list(keys_to_delete))

        return len(keys_to_delete)

    def cached_rag(self, func: Callable) -> Callable:
        """
        Decorator for RAG functions that return (answer, source_chunk_ids).

        @cache.cached_rag
        def ask(query: str) -> tuple[str, list[str]]:
            answer = llm.generate(retrieve(query))
            return answer, ["doc_42_chunk_3"]
        """
        @functools.wraps(func)
        def wrapper(query: str, *args: Any, **kwargs: Any) -> str:
            result = self.lookup(query)
            if result:
                return result.answer

            answer, chunk_ids = func(query, *args, **kwargs)
            self.store(query, answer, chunk_ids)
            return answer

        return wrapper
