import hashlib
import math
import re

from app.domain.ports import EmbeddedGeneratorPort


class SimpleHashEmbeddingGenerator(EmbeddedGeneratorPort):
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        words = self._tokenize(text)

        for word in words:
            index = self._word_to_index(word)
            vector[index] += 1.0

        return self._normalize(vector)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _word_to_index(self, word: str) -> int:
        hash_value = hashlib.md5(word.encode("utf-8")).hexdigest()
        return int(hash_value, 16) % self.dimension

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [value / norm for value in vector]
