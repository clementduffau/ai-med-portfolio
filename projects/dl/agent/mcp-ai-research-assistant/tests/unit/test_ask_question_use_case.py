from app.application.use_cases.ask_question import AskQuestionUseCase
from app.domain.entities import Answer, DocumentChunk, Question, Source
from app.domain.ports import (
    AnswerGeneratorPort,
    DocumentChunkRepositoryPort,
    DocumentRetrieverPort,
)


class FakeAnswerGenerator(AnswerGeneratorPort):
    def generate_answer(
        self,
        question: Question,
        context_chunks: list[DocumentChunk],
    ) -> Answer:
        context = " ".join(chunk.content for chunk in context_chunks)

        return Answer(
            content=f"Answer to '{question.content}' using context: {context}",
            sources=[],
        )


class FakeDocumentRetriever(DocumentRetrieverPort):
    def retrieve(self, question: Question, top_k: int = 3) -> list[Source]:
        return [
            Source(
                document_name="paper.txt",
                chunk_id="paper.txt_chunk_0",
                score=2.0,
            )
        ]


class FakeChunkRepository(DocumentChunkRepositoryPort):
    def __init__(self) -> None:
        self.chunks = [
            DocumentChunk(
                document_name="paper.txt",
                chunk_id="paper.txt_chunk_0",
                content="Transformers use attention.",
            )
        ]

    def save_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.chunks.extend(chunks)

    def list_chunks(self) -> list[DocumentChunk]:
        return self.chunks

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        return [chunk for chunk in self.chunks if chunk.chunk_id in chunk_ids]


def test_ask_question_use_case_returns_answer_with_context_and_sources() -> None:
    question = Question(content="What are transformers?")

    use_case = AskQuestionUseCase(
        answer_generator=FakeAnswerGenerator(),
        document_retriever=FakeDocumentRetriever(),
        chunk_repository=FakeChunkRepository(),
    )

    answer = use_case.execute(question)

    assert answer.content == (
        "Answer to 'What are transformers?' using context: Transformers use attention."
    )
    assert len(answer.sources) == 1
    assert answer.sources[0].document_name == "paper.txt"
    assert answer.sources[0].chunk_id == "paper.txt_chunk_0"
