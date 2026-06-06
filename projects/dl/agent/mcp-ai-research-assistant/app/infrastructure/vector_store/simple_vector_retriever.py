from app.domain.entities import Question, Source
from app.domain.ports import (
    DocumentRetrieverPort,
    EmbeddedGeneratorPort,
    VectorStorePort,
)


class SimpleVectorRetriever(DocumentRetrieverPort):
    def __init__(
        self, embedding_generator: EmbeddedGeneratorPort, vector_store: VectorStorePort
    ) -> None:

        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def retrieve(
        self, question: Question, top_k: int = 3, document_name: str | None = None
    ) -> list[Source]:
        question_embedding = self.embedding_generator.embed(question.content)

        return self.vector_store.search(
            query_embedding=question_embedding, top_k=top_k, document_name=document_name
        )
