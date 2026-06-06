from groq import Groq

from app.domain.entities import DocumentChunk
from app.domain.ports import DocumentSummarizerPort


class GroqDocumentSummarizer(DocumentSummarizerPort):
    def __init__(
        self,
        api_key: str,
        model_name: str,
    ) -> None:
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def summarize(
        self,
        document_name: str,
        chunks: list[DocumentChunk],
    ) -> str:
        context = self._build_context(chunks)

        system_prompt = (
            "Tu es un assistant IA spécialisé dans le résumé de documents. "
            "Tu dois résumer uniquement le contenu fourni. "
            "Réponds en français, de manière claire, structurée et concise."
        )

        user_prompt = f"""
Document :
{document_name}

Contenu :
{context}

Fais un résumé clair en 5 à 8 points maximum.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )

        return response.choices[0].message.content or ""

    def _build_context(self, chunks: list[DocumentChunk]) -> str:
        return "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.content}" for chunk in chunks)
