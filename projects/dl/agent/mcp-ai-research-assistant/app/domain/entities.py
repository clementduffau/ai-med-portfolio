from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    content: str


@dataclass(frozen=True)
class Source:
    document_name: str
    chunk_id: str
    score: float


@dataclass(frozen=True)
class Answer:
    content: str
    sources: list[Source]


@dataclass(frozen=True)
class Document:
    name: str
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    document_name: str
    chunk_id: str
    content: str


@dataclass(frozen=True)
class EmbeddedDocumentChunk:
    chunk: DocumentChunk
    embedding: list[float]
