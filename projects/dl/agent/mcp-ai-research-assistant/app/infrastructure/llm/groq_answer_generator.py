from groq import Groq

from app.domain.entities import Answer, DocumentChunk, Question
from app.domain.exceptions import AnswerGenerationError
from app.domain.ports import AnswerGeneratorPort


class GroqAnswerGenerator(AnswerGeneratorPort):
    def __init__(
        self,
        api_key: str,
        model_name: str,
    ) -> None:
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate_answer(
        self,
        question: Question,
        context_chunks: list[DocumentChunk],
    ) -> Answer:
        if not context_chunks:
            return Answer(
                content="Je n’ai pas trouvé assez de contexte dans les documents pour répondre correctement.",
                sources=[],
            )

        context = self._build_context(context_chunks)

        system_prompt = (
            "Tu es un assistant IA spécialisé dans l'analyse de documents. "
            "Tu dois répondre uniquement à partir du contexte fourni. "
            "Si le contexte ne suffit pas, dis clairement que l'information n'est pas disponible. "
            "Réponds en français, de manière claire et structurée."
        )

        user_prompt = f"""
Contexte :
{context}

Question :
{question.content}
""".strip()

        try:
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
                max_tokens=300,
            )
        except Exception as exc:
            raise AnswerGenerationError("Failed to generate answer with Groq.") from exc

        content = response.choices[0].message.content

        return Answer(
            content=content or "",
            sources=[],
        )

    def _build_context(self, context_chunks: list[DocumentChunk]) -> str:
        return "\n\n".join(
            f"[{chunk.chunk_id}]\n{chunk.content}" for chunk in context_chunks
        )
