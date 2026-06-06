from fastapi.testclient import TestClient

from app.interfaces.api.main import app

client = TestClient(app)


def test_ask_question_returns_answer() -> None:
    response = client.post(
        "/questions/ask",
        json={"question": "What is this document about?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Fake answer to: What is this document about?",
        "sources": [],
    }


def test_ask_question_requires_minimum_length() -> None:
    response = client.post(
        "/questions/ask",
        json={"question": ""},
    )

    assert response.status_code == 422
