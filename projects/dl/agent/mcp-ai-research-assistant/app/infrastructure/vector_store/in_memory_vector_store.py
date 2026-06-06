import math

from app.domain.entities import EmbeddedDocumentChunk, Source
from app.domain.ports import VectorStorePort


class InMemoryDocumentChunkVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._embedded_chunks = []

    def save(self, embedded_chunks: list[EmbeddedDocumentChunk]) -> None:
        self._embedded_chunks.extend(embedded_chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Source]:
        scored_sources = []

        for embedded_chunk in self._embedded_chunks:
            score = self._cosine_similarity(query_embedding, embedded_chunk.embedding)

            if score > 0:
                scored_sources.append(
                    Source(
                        document_name=embedded_chunk.chunk.document_name,
                        chunk_id=embedded_chunk.chunk.chunk_id,
                        score=score,
                    )
                )

        scored_sources.sort(key=lambda x: x.score, reverse=True)

        return scored_sources[:top_k]

    def _cosine_similarity(self, vector_a: list[float], vector_b: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
