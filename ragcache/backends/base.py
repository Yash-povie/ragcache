from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CacheEntry:
    key: str
    query_text: str
    answer: str
    embedding: np.ndarray
    source_chunk_ids: list[str]
    created_at: float
    similarity: float = 0.0


class AbstractCacheBackend(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def vector_search(self, embedding: np.ndarray, top_k: int = 1) -> list[CacheEntry]: ...

    @abstractmethod
    def store(self, entry: CacheEntry) -> None: ...

    @abstractmethod
    def delete(self, keys: list[str]) -> None: ...

    @abstractmethod
    def get_keys_for_chunk(self, chunk_id: str) -> list[str]: ...

    @abstractmethod
    def add_chunk_reverse_index(self, chunk_id: str, cache_key: str) -> None: ...
