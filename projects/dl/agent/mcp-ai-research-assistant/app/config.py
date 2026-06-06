from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MCP AI Research Assistant"
    environment: str = "local"
    debug: bool = True

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "document_chunks"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    groq_api_key: str | None = None
    groq_model_name: str = "llama-3.1-8b-instant"

    mcp_api_base_url: str = "http://127.0.0.1:8000"

    class Config:
        env_file = ".env"


settings = Settings()
