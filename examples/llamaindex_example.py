"""
ragcache + LlamaIndex integration example.

pip install ragcache[sentence-transformers] llama-index
docker-compose up -d
"""

from ragcache import SemanticCache

cache = SemanticCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.92,  # slightly tighter for more precision
)


@cache.cached_rag
def ask(query: str) -> tuple[str, list[str]]:
    """
    The @cache.cached_rag decorator wraps your function automatically.
    Your function must return (answer: str, source_chunk_ids: list[str]).
    """
    # --- Your existing LlamaIndex RAG code ---
    # from llama_index.core import VectorStoreIndex
    # response = index.as_query_engine().query(query)
    # answer = str(response)
    # chunk_ids = [node.node_id for node in response.source_nodes]

    answer = f"LlamaIndex answer to: {query}"
    chunk_ids = ["node_abc123", "node_def456"]
    return answer, chunk_ids


if __name__ == "__main__":
    print(ask("What is the pricing plan?"))
    print(ask("How much does it cost?"))  # should hit cache
