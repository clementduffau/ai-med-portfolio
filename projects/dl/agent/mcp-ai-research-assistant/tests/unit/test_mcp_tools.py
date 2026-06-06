from app.infrastructure.mcp import server


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


def test_list_documents_tool(monkeypatch) -> None:
    def fake_get(url: str, timeout: int):
        assert url.endswith("/documents")
        return FakeResponse(
            {
                "documents": [
                    {
                        "document_name": "paper.txt",
                        "chunks_count": 1,
                    }
                ]
            }
        )

    monkeypatch.setattr(server.requests, "get", fake_get)

    result = server.list_documents()

    assert result["documents"][0]["document_name"] == "paper.txt"
    assert result["documents"][0]["chunks_count"] == 1


def test_ask_question_tool(monkeypatch) -> None:
    def fake_post(url: str, json: dict, timeout: int):
        assert url.endswith("/agent/run")
        assert json["message"] == "Explique le document"
        return FakeResponse(
            {
                "intent": "ask_question",
                "answer": "Réponse test",
                "sources": [],
            }
        )

    monkeypatch.setattr(server.requests, "post", fake_post)

    result = server.ask_question("Explique le document")

    assert result["intent"] == "ask_question"
    assert result["answer"] == "Réponse test"


def test_summarize_document_tool(monkeypatch) -> None:
    def fake_post(url: str, json: dict, timeout: int):
        assert url.endswith("/documents/summarize")
        assert json["document_name"] == "paper.txt"
        return FakeResponse(
            {
                "document_name": "paper.txt",
                "summary": "Résumé test",
                "chunks_used": 1,
            }
        )

    monkeypatch.setattr(server.requests, "post", fake_post)

    result = server.summarize_document("paper.txt")

    assert result["document_name"] == "paper.txt"
    assert result["summary"] == "Résumé test"


def test_delete_document_tool(monkeypatch) -> None:
    def fake_delete(url: str, timeout: int):
        assert url.endswith("/documents/paper.txt")
        return FakeResponse(
            {
                "document_name": "paper.txt",
                "deleted_chunks": 1,
                "status": "deleted",
            }
        )

    monkeypatch.setattr(server.requests, "delete", fake_delete)

    result = server.delete_document("paper.txt")

    assert result["status"] == "deleted"
    assert result["deleted_chunks"] == 1
