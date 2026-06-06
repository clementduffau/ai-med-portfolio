from fastapi import FastAPI

from app.config import settings
from app.interfaces.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Production-ready AI assistant using FastAPI, RAG, MCP and hexagonal architecture.",
    )

    app.include_router(router)

    return app


app = create_app()
