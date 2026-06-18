"""
ragcache + LangChain integration example.

pip install ragcache[sentence-transformers] langchain langchain-openai
docker-compose up -d
"""

from ragcache import SemanticCache

cache = SemanticCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.90,
)


def ask(query: str) -> str:
    result = cache.lookup(query)
    if result:
        print(f"[CACHE HIT] sim={result.similarity:.3f} latency={result.latency_ms:.1f}ms")
        return result.answer

    # --- Your existing LangChain RAG code starts here ---
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI(model="gpt-4o-mini")
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=your_retriever)
    # result_obj = qa_chain.invoke({"query": query})
    # answer = result_obj["result"]
    # source_ids = [doc.metadata["chunk_id"] for doc in result_obj["source_documents"]]

    # placeholder for demo:
    answer = f"LLM answer to: {query}"
    source_ids = ["doc_1_chunk_0"]
    # --- Your existing LangChain RAG code ends here ---

    cache.store(query, answer, source_ids)
    print("[CACHE MISS] stored for future use")
    return answer


if __name__ == "__main__":
    queries = [
        "What is your refund policy?",
        "How do I return an item?",          # semantically similar → should hit cache
        "Can you explain the return process?", # also similar
        "How do I reset my password?",         # different topic → miss
    ]

    for q in queries:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
