"""Retrieve relevant chunks from Milvus by vector similarity."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pymilvus import MilvusClient

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.settings import NaiveRagSettings


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float


class Retriever:
    """Embed a query and search Milvus for the most similar chunks."""

    def __init__(self, encoder: EmbeddingEncoder, milvus_client: MilvusClient, settings: NaiveRagSettings) -> None:
        """Initialize with an embedding encoder, Milvus client, and naive RAG settings."""
        self.encoder = encoder
        self.client = milvus_client
        self.settings = settings

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to query, ranked by similarity score."""
        query_vec = self.encoder.encode_query([query])[0].tolist()

        results = self.client.search(
            collection_name=self.settings.milvus_collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["chunk_id", "text", "source_name", "kind"],
            search_params={
                "metric_type": self.settings.milvus_metric_type,
                "params": {"nprobe": self.settings.milvus_nprobe},
            },
        )

        hits = [
            RetrievedChunk(
                chunk_id=hit["entity"]["chunk_id"],
                text=hit["entity"]["text"],
                source_name=hit["entity"]["source_name"],
                kind=hit["entity"]["kind"],
                score=hit["distance"],
            )
            for hit in results[0]
        ]
        logger.debug("Retrieved {} chunks", len(hits))
        return hits
