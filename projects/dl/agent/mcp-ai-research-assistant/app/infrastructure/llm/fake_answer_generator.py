from app.domain.entities import Answer, DocumentChunk, Question
from app.domain.ports import AnswerGeneratorPort


class FakeAnswerGenerator(AnswerGeneratorPort):
    def generate_answer(
        self, question: Question, context_chunks: list[DocumentChunk]
    ) -> Answer:
        if not context_chunks:
            return Answer(
                content=f"I do not have enough information to answer question : {question.content}",
                sources=[],
            )

        context = "\n".join(chunk.content for chunk in context_chunks)
        return Answer(content=f"Based on the retrieve context : {context}", sources=[])
