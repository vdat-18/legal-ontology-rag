"""Naive RAG pipeline: retrieve then generate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from legal_rag.datastores.milvus import build_milvus_client
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.naive_rag.generator import BamlGenerationClient, generate_answer
from legal_rag.naive_rag.retriever import RetrievedChunk, Retriever, SearchMode
from legal_rag.query_rewriting import SyncQueryRewriter
from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Container for a pipeline response."""

    query: str
    rewritten_query: str | None = None
    answer: str = ""
    sources: tuple[RetrievedChunk, ...] = ()


class NaiveRAGPipeline:
    """End-to-end retrieve-then-generate pipeline."""

    def __init__(
        self,
        settings: Settings,
        milvus_settings: MilvusSettings,
        naive_rag_settings: NaiveRagSettings,
    ) -> None:
        """Initialize the pipeline from application and naive RAG settings."""
        self.settings = naive_rag_settings
        profile = load_configured_embedding_profile(naive_rag_settings.embedding_profile)
        encoder = EmbeddingEncoder.from_profile(profile)
        client = build_milvus_client(milvus_settings)
        self._retriever = Retriever(encoder, client, naive_rag_settings)
        self._generation_client = BamlGenerationClient(settings, naive_rag_settings)
        self._query_rewriter = SyncQueryRewriter(settings)
        logger.info("NaiveRAGPipeline initialized")

    def run(
        self,
        query: str,
        top_k: int | None = None,
        search_mode: str | None = None,
        history: Sequence[tuple[str, str]] = (),
        rewrite_query: bool = True,
    ) -> RAGResult:
        """Retrieve chunks and generate an answer.

        Args:
            query: The user's raw question, possibly context-dependent.
            top_k: Maximum number of chunks to retrieve.
            search_mode: Override the default search mode.
            history: Sequence of (question, answer) pairs from earlier turns.
            rewrite_query: Whether to rewrite the query before retrieval.

        Returns:
            RAGResult with answer and supporting sources.
        """
        k = top_k or self.settings.top_k
        mode = SearchMode(search_mode) if search_mode else None

        rewritten_query = query
        if rewrite_query and history:
            result = self._query_rewriter.rewrite(query, tuple(history))
            rewritten_query = result.rewritten
            if rewritten_query != query:
                logger.debug("Query rewritten: '{}' -> '{}'", query, rewritten_query)

        hits = self._retriever.retrieve(rewritten_query, top_k=k, mode=mode)
        answer = generate_answer(query, hits, self._generation_client)
        active = search_mode or self.settings.search_mode
        logger.info("Pipeline completed: {} sources, {} search", len(hits), active)
        return RAGResult(
            query=query,
            rewritten_query=rewritten_query if rewritten_query != query else None,
            answer=answer,
            sources=tuple(hits),
        )

    def close(self) -> None:
        """Close the underlying retrieval connection."""
        self._retriever.close()
