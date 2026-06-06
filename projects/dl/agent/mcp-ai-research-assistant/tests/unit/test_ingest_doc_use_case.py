from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.domain.entities import Document, DocumentChunk
from app.domain.ports import DocumentChunkerPort, DocumentChunkRepositoryPort


class FakeDocumentChunker(DocumentChunkerPort):
    def chunk(self, document: Document) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                document_name=document.name,
                chunk_id=f"{document.name}_chunk_0",
                content=document.content,
            )
        ]


class FakeChunkRepository(DocumentChunkRepositoryPort):
    def __init__(self) -> None:
        self.saved_chunks: list[DocumentChunk] = []

    def save_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.saved_chunks.extend(chunks)

    def list_chunks(self) -> list[DocumentChunk]:
        return self.saved_chunks


def test_ingest_document_chunks_and_saves_document() -> None:
    document = Document(
        name="paper.txt",
        content="This is a test document.",
    )

    chunker = FakeDocumentChunker()
    repository = FakeChunkRepository()

    use_case = IngestDocumentUseCase(
        document_chunker=chunker,
        chunk_repository=repository,
    )

    chunks = use_case.execute(document)

    assert len(chunks) == 1
    assert repository.saved_chunks == chunks
    assert chunks[0].document_name == "paper.txt"
    assert chunks[0].chunk_id == "paper.txt_chunk_0"
