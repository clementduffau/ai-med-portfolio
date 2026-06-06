class ApplicationError(Exception):
    """Base exception for application errors."""


class AnswerGenerationError(ApplicationError):
    """Raised when the answer generation fails."""


class DocumentIngestionError(ApplicationError):
    """Raised when document ingestion fails."""


class RetrievalError(ApplicationError):
    """Raised when document retrieval fails."""
