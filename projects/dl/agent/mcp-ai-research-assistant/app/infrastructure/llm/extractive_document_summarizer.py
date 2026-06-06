from app.domain.entities import DocumentChunk
from app.domain.ports import DocumentSummarizerPort


class ExtractiveDocumentSummarizer(DocumentSummarizerPort):
    def summarize(
        self,
        document_name: str,
        chunks: list[DocumentChunk],
    ) -> str:
        selected_chunks = chunks[:3]

        bullet_points = [
            f"- {chunk.content.strip()}"
            for chunk in selected_chunks
            if chunk.content.strip()
        ]

        if not bullet_points:
            return "Aucun contenu exploitable trouvé pour ce document."

        return "\n".join(
            [
                f"Résumé extractif du document {document_name} :",
                *bullet_points,
            ]
        )
