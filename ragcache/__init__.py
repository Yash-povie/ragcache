"""
ragcache — semantic caching layer for RAG pipelines.

Drop-in library that puts a Redis vector search cache in front of your LLM.
Similar queries return cached answers in <150ms instead of hitting the LLM every time.

Quickstart:
    pip install ragcache[sentence-transformers]

    from ragcache import SemanticCache

    cache = SemanticCache(redis_url="redis://localhost:6379")

    result = cache.lookup(query)
    if result:
        return result.answer  # cache hit — no LLM call

    answer, chunk_ids = my_rag_pipeline(query)
    cache.store(query, answer, chunk_ids)
    return answer
"""

from .cache import SemanticCache, CacheResult
from .backends import RedisVLBackend, CacheEntry
from .embedders import SentenceTransformerEmbedder, OpenAIEmbedder

__version__ = "0.1.0"
__all__ = [
    "SemanticCache",
    "CacheResult",
    "CacheEntry",
    "RedisVLBackend",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
]
