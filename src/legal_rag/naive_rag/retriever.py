"""Retrieve relevant chunks from Milvus by vector similarity."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pymilvus import MilvusClient

from legal_rag.naive_rag.config import MilvusConfig
from legal_rag.naive_rag.embedder import BGEM3Embedder


@dataclass
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float


class Retriever:
    """Embed a query and search Milvus for the most similar chunks."""

    def __init__(
        self, embedder: BGEM3Embedder, milvus_client: MilvusClient, config: MilvusConfig | None = None
    ) -> None:
        """Initialize with an embedder, Milvus client, and optional config."""
        self.embedder = embedder
        self.client = milvus_client
        self.config = config or MilvusConfig()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Search for the *top_k* chunks most similar to *query*.

        Args:
            query: Natural-language question.
            top_k: Number of results to return.

        Returns:
            Ranked list of retrieved chunks with similarity scores.
        """
        query_vec = self.embedder.encode(query)[0].tolist()

        results = self.client.search(
            collection_name=self.config.collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["chunk_id", "text", "source_name", "kind"],
            search_params={"metric_type": self.config.metric_type, "params": {"nprobe": self.config.nprobe}},
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
        logger.debug("Retrieved {} chunks for query: {:.60s}", len(hits), query)
        return hits
