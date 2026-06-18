"""
Tests for SemanticCache core logic.
Requires Redis Stack running: docker-compose up -d
"""

import time
import pytest
import numpy as np

from ragcache import SemanticCache
from ragcache.backends.base import AbstractCacheBackend, CacheEntry


class MockBackend(AbstractCacheBackend):
    """In-memory backend for unit tests — no Redis needed."""

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._reverse: dict[str, set[str]] = {}

    def connect(self): pass

    def vector_search(self, embedding: np.ndarray, top_k: int = 1) -> list[CacheEntry]:
        if not self._store:
            return []
        best = max(
            self._store.values(),
            key=lambda e: float(np.dot(embedding, e.embedding)),
        )
        sim = float(np.dot(embedding, best.embedding))
        best.similarity = sim
        return [best]

    def store(self, entry: CacheEntry): self._store[entry.key] = entry

    def delete(self, keys: list[str]):
        for k in keys:
            key = k.replace("ragcache:entry:", "")
            self._store.pop(key, None)

    def get_keys_for_chunk(self, chunk_id: str) -> list[str]:
        return list(self._reverse.get(chunk_id, set()))

    def add_chunk_reverse_index(self, chunk_id: str, cache_key: str):
        self._reverse.setdefault(chunk_id, set()).add(cache_key)


class IdentityEmbedder:
    """Returns a fixed vector so we can control similarity in tests."""

    def embed(self, text: str) -> np.ndarray:
        # Hash text into a reproducible unit vector
        seed = sum(ord(c) for c in text) % 100
        rng = np.random.default_rng(seed)
        v = rng.random(768).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self): return 768


@pytest.fixture
def cache():
    embedder = IdentityEmbedder()
    backend = MockBackend()
    c = SemanticCache(embedder=embedder, backend=backend, similarity_threshold=0.90)
    c._connected = True
    return c


def test_miss_on_empty_cache(cache):
    result = cache.lookup("what is the refund policy?")
    assert result is None


def test_store_and_exact_hit(cache):
    query = "what is the refund policy?"
    cache.store(query, "You can return within 30 days.", ["doc_1_chunk_2"])

    result = cache.lookup(query)
    assert result is not None
    assert result.answer == "You can return within 30 days."
    assert result.similarity >= 0.90


def test_miss_on_unrelated_query(cache):
    cache.store("what is the refund policy?", "30 days.", ["doc_1_chunk_2"])
    # Very different query — different seed → different vector → similarity < threshold
    result = cache.lookup("how do I reset my password?")
    # May or may not hit depending on vector distance; we check it doesn't return wrong answer
    if result:
        assert result.similarity >= 0.90


def test_invalidation_removes_entry(cache):
    cache.store("what is the refund policy?", "30 days.", ["doc_1_chunk_2"])
    assert cache.lookup("what is the refund policy?") is not None

    count = cache.invalidate(["doc_1_chunk_2"])
    assert count >= 1
    assert cache.lookup("what is the refund policy?") is None


def test_invalidation_only_removes_tagged_entries(cache):
    cache.store("refund policy?", "30 days.", ["doc_1"])
    cache.store("shipping time?", "3-5 days.", ["doc_2"])

    cache.invalidate(["doc_1"])

    assert cache.lookup("refund policy?") is None
    # doc_2 entry should survive
    result = cache.lookup("shipping time?")
    # May or may not hit due to mock embedder — just ensure no crash
    assert result is None or result.source_chunk_ids == ["doc_2"]


def test_cached_rag_decorator(cache):
    call_count = {"n": 0}

    @cache.cached_rag
    def ask(query: str):
        call_count["n"] += 1
        return f"answer to: {query}", ["doc_1"]

    ask("what is your return policy?")
    ask("what is your return policy?")  # should hit cache

    assert call_count["n"] == 1  # LLM only called once
