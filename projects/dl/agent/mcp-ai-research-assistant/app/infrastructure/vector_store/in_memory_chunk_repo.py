from app.domain.entities import DocumentChunk
from app.domain.ports import DocumentChunkRepositoryPort


class InMemoryDocumentChunkRepository(DocumentChunkRepositoryPort):
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []

    def save_chunks(self, chunks: list[DocumentChunk]) -> None:
        self._chunks.extend(chunks)

    def list_chunks(self) -> list[DocumentChunk]:
        return self._chunks

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        chunk_ids_set = set(chunk_ids)
        return [chunk for chunk in self._chunks if chunk.chunk_id in chunk_ids_set]
