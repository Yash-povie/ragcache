import numpy as np
from typing import Optional

from .base import AbstractEmbedder

_DEFAULT_MODEL = "all-mpnet-base-v2"


class SentenceTransformerEmbedder(AbstractEmbedder):
    """
    Zero-API-cost embedder using sentence-transformers.
    Default model: all-mpnet-base-v2 (768 dims, best speed/accuracy balance).
    Swap to all-MiniLM-L6-v2 (384 dims) for higher throughput at slightly lower accuracy.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._dims: Optional[int] = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install ragcache[sentence-transformers]"
                )
            self._model = SentenceTransformer(self._model_name)
            self._dims = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        return [v.astype(np.float32) for v in vecs]

    @property
    def dimensions(self) -> int:
        self._load()
        return self._dims
