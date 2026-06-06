from abc import ABC, abstractmethod

from app.domain.entities import (
    Answer,
    Document,
    DocumentChunk,
    EmbeddedDocumentChunk,
    Question,
    Source,
)


class AnswerGeneratorPort(ABC):
    @abstractmethod
    def generate_answer(
        self, question: Question, context_chunks: list[DocumentChunk]
    ) -> Answer:
        pass


class DocumentChunkerPort(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[DocumentChunk]:
        pass


class DocumentChunkRepositoryPort(ABC):
    @abstractmethod
    def save_chunks(self, chunks: list[DocumentChunk]) -> None:
        pass

    @abstractmethod
    def list_chunks(self) -> list[DocumentChunk]:
        pass

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        pass


class DocumentRetrieverPort(ABC):
    @abstractmethod
    def retrieve(
        self, quesiton: Question, top_k: int = 3, document_name: str | None = None
    ) -> list[Source]:
        pass


class EmbeddedGeneratorPort(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass


class VectorStorePort(ABC):
    @abstractmethod
    def save(self, embedded_chunks: list[EmbeddedDocumentChunk]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embeddding: list[float],
        top_k: int = 3,
        document_name: str | None = None,
    ) -> list[Source]:
        pass

    @abstractmethod
    def list_chunks(self) -> list[DocumentChunk]:
        pass

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        pass

    @abstractmethod
    def get_chunks_by_document_name(self, document_name: str) -> list[DocumentChunk]:
        pass

    @abstractmethod
    def list_document_names(self) -> dict[str, int]:
        pass

    @abstractmethod
    def delete_chunks_by_document_name(self, document_name: str) -> int:
        pass


class DocumentSummarizerPort(ABC):
    @abstractmethod
    def summarize(
        self,
        document_name: str,
        chunks: list[DocumentChunk],
    ) -> str:
        pass
