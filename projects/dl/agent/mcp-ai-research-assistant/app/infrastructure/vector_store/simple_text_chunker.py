from app.domain.entities import Document, DocumentChunk
from app.domain.ports import DocumentChunkerPort


class SimpleTextChunker(DocumentChunkerPort):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks = []
        step = self.chunk_size - self.chunk_overlap

        for index, start in enumerate(range(0, len(document.content), step)):
            chunk_content = document.content[start : start + self.chunk_size]

            if not chunk_content:
                break

            chunks.append(
                DocumentChunk(
                    document_name=document.name,
                    chunk_id=f"{document.name}_chunk_{index}",
                    content=chunk_content,
                )
            )

            if start + self.chunk_size >= len(document.content):
                break

        return chunks
