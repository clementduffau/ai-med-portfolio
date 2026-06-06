from app.domain.entities import DocumentChunk
from app.domain.ports import DocumentSummarizerPort, VectorStorePort


class SummarizeDocumentUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        document_summarizer: DocumentSummarizerPort,
    ) -> None:
        self.vector_store = vector_store
        self.document_summarizer = document_summarizer

    def execute(self, document_name: str) -> tuple[str, list[DocumentChunk]]:
        chunks = self.vector_store.get_chunks_by_document_name(document_name)

        if not chunks:
            return (
                "Aucun document correspondant n’a été trouvé.",
                [],
            )

        summary = self.document_summarizer.summarize(
            document_name=document_name,
            chunks=chunks,
        )

        return summary, chunks
