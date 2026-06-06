from fastapi.testclient import TestClient

from app.interfaces.api.main import app

client = TestClient(app)


def test_ingest_document_returns_chunks() -> None:
    response = client.post(
        "/documents/ingest",
        json={
            "name": "paper.txt",
            "content": "This is a test document.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_name"] == "paper.txt"
    assert data["status"] == "ingested"
    assert data["chunks_count"] == 1
    assert data["chunks"][0]["chunk_id"] == "paper.txt_chunk_0"
    assert data["chunks"][0]["content"] == "This is a test document."


def test_ingest_document_requires_content() -> None:
    response = client.post(
        "/documents/ingest",
        json={
            "name": "paper.txt",
            "content": "",
        },
    )

    assert response.status_code == 422
