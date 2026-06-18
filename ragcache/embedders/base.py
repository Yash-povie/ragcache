from abc import ABC, abstractmethod
import numpy as np


class AbstractEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...
