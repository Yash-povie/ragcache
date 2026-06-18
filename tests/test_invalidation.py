"""
Invalidation webhook tests.
Uses httpx TestClient — no live server needed.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragcache import SemanticCache
from ragcache.invalidation.webhook import build_invalidation_router
from tests.test_cache import MockBackend, IdentityEmbedder


@pytest.fixture
def app_and_cache():
    embedder = IdentityEmbedder()
    backend = MockBackend()
    cache = SemanticCache(embedder=embedder, backend=backend, similarity_threshold=0.90)
    cache._connected = True

    app = FastAPI()
    app.include_router(build_invalidation_router(cache), prefix="/ragcache")
    return app, cache


def test_health_endpoint(app_and_cache):
    app, _ = app_and_cache
    client = TestClient(app)
    resp = client.get("/ragcache/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_invalidate_endpoint(app_and_cache):
    app, cache = app_and_cache
    cache.store("refund policy?", "30 days.", ["doc_42_chunk_3"])

    client = TestClient(app)
    resp = client.post("/ragcache/invalidate", json={"chunk_ids": ["doc_42_chunk_3"]})

    assert resp.status_code == 200
    data = resp.json()
    assert data["invalidated"] >= 1
    assert "doc_42_chunk_3" in data["chunk_ids"]


def test_invalidate_nonexistent_chunk_is_noop(app_and_cache):
    app, _ = app_and_cache
    client = TestClient(app)
    resp = client.post("/ragcache/invalidate", json={"chunk_ids": ["chunk_that_doesnt_exist"]})
    assert resp.status_code == 200
    assert resp.json()["invalidated"] == 0
