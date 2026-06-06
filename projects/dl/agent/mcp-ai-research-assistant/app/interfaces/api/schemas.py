from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class AskQuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    document_name: str | None = None


class SourceResponse(BaseModel):
    document_name: str
    chunk_id: str
    score: float


class AskQuestionResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class IngestDocumentRequest(BaseModel):
    name: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class DocumentChunkResponse(BaseModel):
    chunk_id: str
    content: str
    content_length: int


class IngestDocumentResponse(BaseModel):
    document_name: str
    status: str
    chunks_count: int
    chunks: list[DocumentChunkResponse]


class SummarizeDocumentRequest(BaseModel):
    document_name: str = Field(..., min_length=1)


class SummarizeDocumentResponse(BaseModel):
    document_name: str
    summary: str
    chunks_used: int


class DocumentInfoResponse(BaseModel):
    document_name: str
    chunks_count: int


class ListDocumentsResponse(BaseModel):
    documents: list[DocumentInfoResponse]


class DeleteDocumentResponse(BaseModel):
    document_name: str
    deleted_chunks: int


class AgentRunRequest(BaseModel):
    message: str = Field(..., min_length=3)


class AgentRunResponse(BaseModel):
    intent: str
    answer: str
    sources: list[SourceResponse]


class AgentToolResponse(BaseModel):
    name: str
    description: str
    input_example: dict


class AgentToolsResponse(BaseModel):
    tools: list[AgentToolResponse]
