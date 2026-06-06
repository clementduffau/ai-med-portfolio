from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.application.agents.document_agent import DocumentAgent
from app.application.use_cases.ask_question import AskQuestionUseCase
from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.application.use_cases.summarize_document import SummarizeDocumentUseCase
from app.domain.entities import Document, Question
from app.domain.exceptions import (
    AnswerGenerationError,
    DocumentIngestionError,
    RetrievalError,
)
from app.domain.ports import VectorStorePort
from app.infrastructure.document_loader.file_text_extractor import FileTextExtractor
from app.interfaces.api.dependencies import (
    get_ask_question_use_case,
    get_document_agent,
    get_summarize_document_use_case,
    get_vector_store,
    ingest_document_use_case,
)
from app.interfaces.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentToolResponse,
    AgentToolsResponse,
    AskQuestionRequest,
    AskQuestionResponse,
    DeleteDocumentResponse,
    DocumentChunkResponse,
    DocumentInfoResponse,
    HealthResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    ListDocumentsResponse,
    SourceResponse,
    SummarizeDocumentRequest,
    SummarizeDocumentResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="mcp-ai-research-assistant",
    )


@router.post("/questions/ask", response_model=AskQuestionResponse)
def ask_question(
    request: AskQuestionRequest,
    use_case: AskQuestionUseCase = Depends(get_ask_question_use_case),
) -> AskQuestionResponse:
    question = Question(content=request.question)

    try:
        answer = use_case.execute(question)
    except AnswerGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail="The LLM provider failed to generate an answer.",
        ) from exc
    except RetrievalError as exc:
        raise HTTPException(
            status_code=502,
            detail="The retrieval system failed.",
        ) from exc

    sources = [
        SourceResponse(
            document_name=source.document_name,
            chunk_id=source.chunk_id,
            score=source.score,
        )
        for source in answer.sources
    ]

    return AskQuestionResponse(
        answer=answer.content,
        sources=sources,
    )


@router.post("/documents/ingest", response_model=IngestDocumentResponse)
def ingest_document(
    request: IngestDocumentRequest,
    use_case: IngestDocumentUseCase = Depends(ingest_document_use_case),
) -> IngestDocumentResponse:
    document = Document(
        name=request.name,
        content=request.content,
    )

    try:
        chunks = use_case.execute(document)
    except DocumentIngestionError as exc:
        raise HTTPException(
            status_code=500,
            detail="The document ingestion failed.",
        ) from exc

    chunk_responses = [
        DocumentChunkResponse(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            content_length=len(chunk.content),
        )
        for chunk in chunks
    ]

    return IngestDocumentResponse(
        document_name=document.name,
        status="ingested",
        chunks_count=len(chunks),
        chunks=chunk_responses,
    )


@router.get("/documents/chunks", response_model=list[DocumentChunkResponse])
def list_document_chunks(
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> list[DocumentChunkResponse]:
    chunks = vector_store.list_chunks()

    return [
        DocumentChunkResponse(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            content_length=len(chunk.content),
        )
        for chunk in chunks
    ]


@router.post("/documents/summarize", response_model=SummarizeDocumentResponse)
def summarize_document(
    request: SummarizeDocumentRequest,
    use_case: SummarizeDocumentUseCase = Depends(get_summarize_document_use_case),
) -> SummarizeDocumentResponse:
    summary, chunks = use_case.execute(request.document_name)

    return SummarizeDocumentResponse(
        document_name=request.document_name,
        summary=summary,
        chunks_used=len(chunks),
    )


@router.post("/documents/upload", response_model=IngestDocumentResponse)
def upload_document(
    file: UploadFile = File(...),
    use_case: IngestDocumentUseCase = Depends(ingest_document_use_case),
) -> IngestDocumentUseCase:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must have a filename.",
        )

    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and pdf files are supported",
        )

    raw_content = file.file.read()
    extractor = FileTextExtractor()

    try:
        content = extractor.extract_text(filename=file.filename, content=raw_content)

    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode the file content. ensure it is a valid utf8 text file",
        ) from exc

    document = Document(
        name=file.filename,
        content=content,
    )

    try:
        chunks = use_case.execute(document)
    except DocumentIngestionError as exc:
        raise HTTPException(
            status_code=500,
            detail="The document ingestion failed.",
        ) from exc

    chunk_responses = [
        DocumentChunkResponse(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            content_length=len(chunk.content),
        )
        for chunk in chunks
    ]

    return IngestDocumentResponse(
        document_name=document.name,
        status="ingested",
        chunks_count=len(chunks),
        chunks=chunk_responses,
    )


@router.get("/documents", response_model=ListDocumentsResponse)
def list_documents(
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> ListDocumentsResponse:

    documents = vector_store.list_document_names()

    return ListDocumentsResponse(
        documents=[
            DocumentInfoResponse(
                document_name=document_name,
                chunks_count=chunks_count,
            )
            for document_name, chunks_count in documents.items()
        ]
    )


@router.delete("/documents/{document_name}", response_model=DeleteDocumentResponse)
def delete_document(
    document_name: str,
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> DeleteDocumentResponse:

    deleted_chunks = vector_store.delete_chunks_by_document_name(document_name)

    if deleted_chunks == 0:
        return DeleteDocumentResponse(
            document_name=document_name,
            deleted_chunks=0,
            status="not_found",
        )

    return DeleteDocumentResponse(
        document_name=document_name,
        deleted_chunks=deleted_chunks,
        status="deleted",
    )


@router.post("/agent/run", response_model=AgentRunResponse)
def run_agent(
    request: AgentRunRequest,
    agent: DocumentAgent = Depends(get_document_agent),
) -> AgentRunResponse:

    try:
        result = agent.run(request.message)

        return AgentRunResponse(
            intent=result["intent"],
            answer=result["answer"],
            sources=[
                SourceResponse(
                    document_name=source["document_name"],
                    chunk_id=source["chunk_id"],
                    score=source["score"],
                )
                for source in result["sources"]
            ],
        )

    except Exception:
        return AgentRunResponse(
            intent="error",
            answer=(
                "L’agent n’a pas pu exécuter la demande. "
                "Vérifie qu’un document a bien été uploadé et que Qdrant/Groq sont disponibles."
            ),
            sources=[],
        )


@router.get("/agent/tools", response_model=AgentToolsResponse)
def list_agent_tools() -> AgentToolsResponse:
    return AgentToolsResponse(
        tools=[
            AgentToolResponse(
                name="list_documents",
                description="List all documents currently available in the vector database.",
                input_example={"message": "Liste les documents disponibles"},
            ),
            AgentToolResponse(
                name="ask_question",
                description="Ask a question over all documents or over a specific document.",
                input_example={
                    "message": "Explique la méthode des transformers dans paper.txt"
                },
            ),
            AgentToolResponse(
                name="summarize_document",
                description="Summarize a specific uploaded document.",
                input_example={"message": "Résume le document paper.txt"},
            ),
            AgentToolResponse(
                name="unknown",
                description="Fallback when the agent cannot understand the requested action.",
                input_example={"message": "Fais quelque chose"},
            ),
        ]
    )
