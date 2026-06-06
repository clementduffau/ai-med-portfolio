import requests
from mcp.server.fastmcp import FastMCP

from app.config import settings

mcp = FastMCP("mcp-ai-research-assistant")

API_BASE_URL = settings.mcp_api_base_url


@mcp.tool()
def list_documents() -> dict:
    """List all documents currently available in the vector database."""
    response = requests.get(f"{API_BASE_URL}/documents", timeout=30)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def ask_question(message: str) -> dict:
    """Ask a natural language question over the uploaded documents."""
    response = requests.post(
        f"{API_BASE_URL}/agent/run",
        json={"message": message},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def summarize_document(document_name: str) -> dict:
    """Summarize one uploaded document by name."""
    response = requests.post(
        f"{API_BASE_URL}/documents/summarize",
        json={"document_name": document_name},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def delete_document(document_name: str) -> dict:
    """Delete one uploaded document and all its chunks."""
    response = requests.delete(
        f"{API_BASE_URL}/documents/{document_name}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run()
