from sentence_transformers import SentenceTransformer

from app.domain.ports import EmbeddedGeneratorPort


class SentenceTransformerEmbeddingGenerator(EmbeddedGeneratorPort):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

    def embed(self, text: str) -> list[float]:
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        return self.model
