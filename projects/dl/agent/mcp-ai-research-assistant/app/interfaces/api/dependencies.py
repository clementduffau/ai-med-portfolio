from app.application.agents.document_agent import DocumentAgent
from app.application.use_cases.ask_question import AskQuestionUseCase
from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.application.use_cases.summarize_document import SummarizeDocumentUseCase
from app.config import settings
from app.infrastructure.llm.extractive_document_summarizer import (
    ExtractiveDocumentSummarizer,
)
from app.infrastructure.llm.fake_answer_generator import FakeAnswerGenerator
from app.infrastructure.llm.groq_answer_generator import GroqAnswerGenerator
from app.infrastructure.llm.groq_document_summarizer import GroqDocumentSummarizer
from app.infrastructure.vector_store.in_memory_chunk_repo import (
    InMemoryDocumentChunkRepository,
)
from app.infrastructure.vector_store.qdrant_vector_store import (
    QdrantDocumentChunkVectorStore,
)
from app.infrastructure.vector_store.sentence_transformer_embedding_generator import (
    SentenceTransformerEmbeddingGenerator,
)
from app.infrastructure.vector_store.simple_text_chunker import SimpleTextChunker
from app.infrastructure.vector_store.simple_vector_retriever import (
    SimpleVectorRetriever,
)

chunk_repository = InMemoryDocumentChunkRepository()
embedding_generator = SentenceTransformerEmbeddingGenerator(
    model_name=settings.embedding_model_name
)
vector_store = QdrantDocumentChunkVectorStore(
    url=settings.qdrant_url,
    collection_name=settings.qdrant_collection_name,
    vector_size=settings.embedding_dimension,
)


def get_ask_question_use_case() -> AskQuestionUseCase:
    if settings.groq_api_key is None:
        answer_generator = FakeAnswerGenerator()
    else:
        answer_generator = GroqAnswerGenerator(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
        )

    document_retriever = SimpleVectorRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
    )

    return AskQuestionUseCase(
        answer_generator=answer_generator,
        document_retriever=document_retriever,
        vector_store=vector_store,
    )


def ingest_document_use_case() -> IngestDocumentUseCase:
    document_chunker = SimpleTextChunker(chunk_size=200, chunk_overlap=50)
    return IngestDocumentUseCase(
        document_chunker=document_chunker,
        embedding_generator=embedding_generator,
        vector_store=vector_store,
    )


def get_vector_store() -> QdrantDocumentChunkVectorStore:
    return vector_store


def get_summarize_document_use_case() -> SummarizeDocumentUseCase:
    if settings.groq_api_key is None:
        document_summarizer = ExtractiveDocumentSummarizer()
    else:
        document_summarizer = GroqDocumentSummarizer(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
        )

    return SummarizeDocumentUseCase(
        vector_store=vector_store,
        document_summarizer=document_summarizer,
    )


def get_document_agent() -> DocumentAgent:
    return DocumentAgent(
        ask_question_use_case=get_ask_question_use_case(),
        summarize_document_use_case=get_summarize_document_use_case(),
        vector_store=vector_store,
    )
