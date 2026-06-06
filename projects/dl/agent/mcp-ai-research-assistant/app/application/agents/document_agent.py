from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.application.use_cases.ask_question import AskQuestionUseCase
from app.application.use_cases.summarize_document import SummarizeDocumentUseCase
from app.domain.entities import Question
from app.domain.ports import VectorStorePort

Intent = Literal[
    "ask_question",
    "summarize_document",
    "list_documents",
    "unknown",
]


class DocumentAgentState(TypedDict):
    message: str
    intent: Intent
    document_name: str | None
    answer: str
    sources: list[dict]


class DocumentAgent:
    def __init__(
        self,
        ask_question_use_case: AskQuestionUseCase,
        summarize_document_use_case: SummarizeDocumentUseCase,
        vector_store: VectorStorePort,
    ) -> None:
        self.ask_question_use_case = ask_question_use_case
        self.summarize_document_use_case = summarize_document_use_case
        self.vector_store = vector_store
        self.graph = self._build_graph()

    def run(self, message: str) -> DocumentAgentState:
        initial_state: DocumentAgentState = {
            "message": message,
            "intent": "unknown",
            "document_name": None,
            "answer": "",
            "sources": [],
        }

        return self.graph.invoke(initial_state)

    def _build_graph(self):
        graph = StateGraph(DocumentAgentState)

        graph.add_node("detect_intent", self._detect_intent)
        graph.add_node("ask_question", self._ask_question)
        graph.add_node("summarize_document", self._summarize_document)
        graph.add_node("list_documents", self._list_documents)
        graph.add_node("unknown", self._unknown)

        graph.add_edge(START, "detect_intent")

        graph.add_conditional_edges(
            "detect_intent",
            self._route_by_intent,
            {
                "ask_question": "ask_question",
                "summarize_document": "summarize_document",
                "list_documents": "list_documents",
                "unknown": "unknown",
            },
        )

        graph.add_edge("ask_question", END)
        graph.add_edge("summarize_document", END)
        graph.add_edge("list_documents", END)
        graph.add_edge("unknown", END)

        return graph.compile()

    def _detect_intent(self, state: DocumentAgentState) -> dict:
        raw_message = state["message"]
        message = raw_message.lower()
        document_name = self._extract_document_name(raw_message)

        if any(word in message for word in ["liste", "list", "documents disponibles"]):
            intent: Intent = "list_documents"
        elif any(
            word in message for word in ["résume", "resume", "summary", "summarize"]
        ):
            intent = "summarize_document"
        elif "?" in message or any(
            word in message
            for word in ["explique", "explain", "quoi", "what", "comment", "how"]
        ):
            intent = "ask_question"
        else:
            intent = "unknown"

        return {"intent": intent, "document_name": document_name}

    def _route_by_intent(self, state: DocumentAgentState) -> Intent:
        return state["intent"]

    def _ask_question(self, state: DocumentAgentState) -> dict:
        documents = self.vector_store.list_document_names()

        if not documents:
            return {
                "answer": (
                    "Aucun document n’est disponible pour le moment. "
                    "Upload d’abord un document avec /documents/upload."
                ),
                "sources": [],
            }

        question = Question(content=state["message"])

        answer = self.ask_question_use_case.execute(
            question,
            document_name=state["document_name"],
        )

        if not answer.sources:
            return {
                "answer": (
                    "Je n’ai pas trouvé de passage pertinent dans les documents pour répondre à cette question."
                ),
                "sources": [],
            }

        return {
            "answer": answer.content,
            "sources": [
                {
                    "document_name": source.document_name,
                    "chunk_id": source.chunk_id,
                    "score": source.score,
                }
                for source in answer.sources
            ],
        }

    def _summarize_document(self, state: DocumentAgentState) -> dict:
        document_name = state["document_name"]

        if document_name is None:
            documents = self.vector_store.list_document_names()

            if not documents:
                return {
                    "answer": (
                        "Aucun document n’est disponible pour le moment. "
                        "Upload d’abord un document avant de demander un résumé."
                    ),
                    "sources": [],
                }

            available_documents = ", ".join(documents.keys())

            return {
                "answer": (
                    "Je dois connaître le nom du document à résumer. "
                    f"Documents disponibles : {available_documents}"
                ),
                "sources": [],
            }

        summary, chunks = self.summarize_document_use_case.execute(document_name)
        return {
            "answer": summary,
            "sources": [
                {
                    "document_name": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                    "score": 1.0,
                }
                for chunk in chunks
            ],
        }

    def _list_documents(self, state: DocumentAgentState) -> dict:
        documents = self.vector_store.list_document_names()

        if not documents:
            return {"answer": "aucun doc dispo", "sources": []}

        lines = [
            f"- {document_name} ({chunks_count} chunks)"
            for document_name, chunks_count in documents.items()
        ]

        return {
            "answer": "Documents disponibles :\n" + "\n".join(lines),
            "sources": [],
        }

    def _unknown(self, state: DocumentAgentState) -> dict:
        return {
            "answer": (
                "Je n’ai pas compris l’action demandée. "
                "Tu peux me demander de lister les documents, résumer un document, "
                "ou poser une question sur les documents."
            ),
            "sources": [],
        }

    def _extract_document_name(self, message: str) -> str | None:
        words = message.split()

        for word in words:
            cleaned_word = word.strip(".,;:!?()[]{}\"'")

            if cleaned_word.endswith((".txt", ".pdf")):
                return cleaned_word

        return None
