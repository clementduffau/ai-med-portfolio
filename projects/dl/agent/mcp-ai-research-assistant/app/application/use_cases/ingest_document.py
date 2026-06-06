from app.domain.entities import Document, DocumentChunk, EmbeddedDocumentChunk
from app.domain.exceptions import DocumentIngestionError
from app.domain.ports import DocumentChunkerPort, EmbeddedGeneratorPort, VectorStorePort


class IngestDocumentUseCase:
    def __init__(
        self,
        document_chunker: DocumentChunkerPort,
        embedding_generator: EmbeddedGeneratorPort,
        vector_store: VectorStorePort,
    ) -> None:
        self.document_chunker = document_chunker
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def execute(self, document: Document) -> list[DocumentChunk]:
        try:
            chunks = self.document_chunker.chunk(document)

            embedded_chunks = [
                EmbeddedDocumentChunk(
                    chunk=chunk,
                    embedding=self.embedding_generator.embed(chunk.content),
                )
                for chunk in chunks
            ]

            self.vector_store.save(embedded_chunks)

            return chunks
        except Exception as exc:
            raise DocumentIngestionError(
                f"Failed to ingest document: {document.name}"
            ) from exc
