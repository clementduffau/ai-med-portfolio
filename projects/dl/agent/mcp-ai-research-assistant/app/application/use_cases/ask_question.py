from app.domain.entities import Answer, Question
from app.domain.exceptions import RetrievalError
from app.domain.ports import AnswerGeneratorPort, DocumentRetrieverPort, VectorStorePort


class AskQuestionUseCase:
    def __init__(
        self,
        answer_generator: AnswerGeneratorPort,
        document_retriever: DocumentRetrieverPort,
        vector_store: VectorStorePort,
    ) -> None:

        self.answer_generator = answer_generator
        self.vector_store = vector_store
        self.document_retriever = document_retriever

    def execute(self, question: Question, document_name: str | None = None) -> Answer:
        try:
            sources = self.document_retriever.retrieve(
                question, document_name=document_name
            )
            chunk_ids = [source.chunk_id for source in sources]
            context_chunks = self.vector_store.get_chunks_by_ids(chunk_ids)
        except Exception as exc:
            raise RetrievalError("Failed to retrieve relevant chunks.") from exc

        answer = self.answer_generator.generate_answer(
            question=question,
            context_chunks=context_chunks,
        )

        return Answer(
            content=answer.content,
            sources=sources,
        )
