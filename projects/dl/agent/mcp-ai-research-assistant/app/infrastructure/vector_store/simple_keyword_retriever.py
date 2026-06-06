import re

from app.domain.entities import Question, Source
from app.domain.ports import DocumentChunkRepositoryPort, DocumentRetrieverPort


class SimpleKeywordRetriever(DocumentRetrieverPort):
    def __init__(self, chunk_repository: DocumentChunkRepositoryPort) -> None:
        self.chunk_repository = chunk_repository

    def retrieve(self, question: Question, top_k: int = 3) -> list[Source]:
        chunks = self.chunk_repository.list_chunks()
        question_words = self._tokenize(question.content)

        scored_source = []

        for chunk in chunks:
            chunk_words = self._tokenize(chunk.content)
            common_words = question_words.intersection(chunk_words)

            score = len(common_words)

            if score > 0:
                scored_source.append(
                    Source(
                        document_name=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        score=float(score),
                    )
                )
        scored_source.sort(key=lambda source: source.score, reverse=True)

        return scored_source[:top_k]

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return set(words)
