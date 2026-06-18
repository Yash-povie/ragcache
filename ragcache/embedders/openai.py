import numpy as np
from .base import AbstractEmbedder

_DEFAULT_MODEL = "text-embedding-3-small"
_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}


class OpenAIEmbedder(AbstractEmbedder):
    """OpenAI embeddings — costs per call but higher accuracy for ambiguous queries."""

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _load(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai is not installed. Run: pip install ragcache[openai]"
                )
            self._client = OpenAI(api_key=self._api_key)

    def embed(self, text: str) -> np.ndarray:
        self._load()
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._load()
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    @property
    def dimensions(self) -> int:
        return _DIMS.get(self._model, 1536)
