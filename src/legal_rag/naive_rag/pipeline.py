"""Naive RAG pipeline: retrieve then generate."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from pymilvus import MilvusClient

from legal_rag.naive_rag.config import NaiveRAGConfig
from legal_rag.naive_rag.embedder import BGEM3Embedder
from legal_rag.naive_rag.generator import generate
from legal_rag.naive_rag.retriever import RetrievedChunk, Retriever


@dataclass
class RAGResult:
    """Container for a pipeline response."""

    query: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


class NaiveRAGPipeline:
    """End-to-end retrieve-then-generate pipeline."""

    def __init__(self, config: NaiveRAGConfig | None = None) -> None:
        """Initialize the pipeline with the given configuration."""
        self.config = config or NaiveRAGConfig()
        self._embedder = BGEM3Embedder(self.config.embedder)
        self._client = MilvusClient(uri=self.config.milvus.uri)
        self._retriever = Retriever(self._embedder, self._client, self.config.milvus)
        logger.info("NaiveRAGPipeline initialized")

    async def run(self, query: str, top_k: int | None = None) -> RAGResult:
        """Execute the full pipeline for a single query.

        Args:
            query: Natural-language question.
            top_k: Override the default number of retrieved chunks.

        Returns:
            Result containing the answer and source chunks.
        """
        k = top_k or self.config.top_k
        hits = self._retriever.retrieve(query, top_k=k)
        answer = await generate(query, hits, self.config.generator)
        logger.info("Pipeline completed for: {:.60s}", query)
        return RAGResult(query=query, answer=answer, sources=hits)
