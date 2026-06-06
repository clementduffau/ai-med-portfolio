import pytest

from app.domain.entities import Answer, Document, Question, Source


def test_create_question() -> None:
    question = Question(content="what is this document about ?")
    assert question.content == "What is this document about?"


def test_create_document() -> None:
    document = Document(name="paper.pdf", content="this is the content of the document")
    assert document.name == "What is this document about?"
    assert document.content == "this is the of the document"


def test_create_answer_with_sources() -> None:
    source = Source(document_name="paper.pdf", chunk_id="chunk_1", score=0.91)

    answer = Answer(
        content="This document is about AI",
        sources=[source],
    )

    assert answer.content == "This document is about AI."
    assert answer.sources[0].document_name == "paper.pdf"
    assert answer.sources[0].score == 0.91


def test_question_is_immutable() -> None:
    question = Question(content="Initial question")

    with pytest.raises(Exception):
        question.content = "Modified question"
