"""
FastAPI invalidation webhook — mount this into any existing FastAPI app.

    from fastapi import FastAPI
    from ragcache import SemanticCache
    from ragcache.invalidation.webhook import build_invalidation_router

    app = FastAPI()
    cache = SemanticCache(redis_url="redis://localhost:6379")
    app.include_router(build_invalidation_router(cache), prefix="/ragcache")

Then when a document chunk is updated, POST to /ragcache/invalidate:
    {"chunk_ids": ["doc_42_chunk_3", "doc_17_chunk_1"]}
"""

from fastapi import APIRouter
from pydantic import BaseModel

from ragcache.cache import SemanticCache


class InvalidationRequest(BaseModel):
    chunk_ids: list[str]


class InvalidationResponse(BaseModel):
    invalidated: int
    chunk_ids: list[str]


def build_invalidation_router(cache: SemanticCache) -> APIRouter:
    router = APIRouter()

    @router.post("/invalidate", response_model=InvalidationResponse)
    async def invalidate(req: InvalidationRequest) -> InvalidationResponse:
        count = cache.invalidate(req.chunk_ids)
        return InvalidationResponse(invalidated=count, chunk_ids=req.chunk_ids)

    @router.get("/health")
    async def health():
        return {"status": "ok"}

    return router
