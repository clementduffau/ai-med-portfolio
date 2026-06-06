import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from app.domain.entities import DocumentChunk, EmbeddedDocumentChunk, Source
from app.domain.ports import VectorStorePort


class QdrantDocumentChunkVectorStore(VectorStorePort):
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size

        self._ensure_collection_exists()

    def save(self, embedded_chunks: list[EmbeddedDocumentChunk]) -> None:
        points = [
            PointStruct(
                id=self._build_point_id(embedded_chunk.chunk.chunk_id),
                vector=embedded_chunk.embedding,
                payload={
                    "document_name": embedded_chunk.chunk.document_name,
                    "chunk_id": embedded_chunk.chunk.chunk_id,
                    "content": embedded_chunk.chunk.content,
                },
            )
            for embedded_chunk in embedded_chunks
        ]

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        document_name: str | None = None,
    ) -> list[Source]:
        query_filter = None

        if document_name is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            Source(
                document_name=str(point.payload["document_name"]),
                chunk_id=str(point.payload["chunk_id"]),
                score=float(point.score),
            )
            for point in results.points
            if point.payload is not None
        ]

    def list_chunks(self) -> list[DocumentChunk]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self._point_to_document_chunk(point)
            for point in points
            if point.payload is not None
        ]

    def get_chunks_by_ids(self, chunk_ids):
        point_ids = [self._build_point_id(chunk_id) for chunk_id in chunk_ids]

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self._point_to_document_chunk(point)
            for point in points
            if point.payload is not None
        ]

    def get_chunks_by_document_name(self, document_name: str) -> list[DocumentChunk]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "document_name",
                        "match": {
                            "value": document_name,
                        },
                    }
                ]
            },
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self._point_to_document_chunk(point)
            for point in points
            if point.payload is not None
        ]

    def list_document_names(self) -> dict[str, int]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        documents: dict[str, int] = {}

        for point in points:
            if point.payload is None:
                continue

            document_name = str(point.payload["document_name"])

            if document_name not in documents:
                documents[document_name] = 0

            documents[document_name] += 1

        return documents

    def delete_chunks_by_document_name(self, document_name: str) -> int:
        chunks = self.get_chunks_by_document_name(document_name)

        if not chunks:
            return 0

        point_ids = [self._build_point_id(chunk.chunk_id) for chunk in chunks]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=point_ids,
            ),
        )

        return len(point_ids)

    def _ensure_collection_exists(self) -> None:
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def _build_point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def _point_to_document_chunk(self, point) -> DocumentChunk:
        payload = point.payload

        return DocumentChunk(
            document_name=str(payload["document_name"]),
            chunk_id=str(payload["chunk_id"]),
            content=str(payload["content"]),
        )
