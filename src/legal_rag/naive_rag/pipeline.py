"""Naive RAG pipeline: retrieve then generate."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from legal_rag.datastores.milvus import build_milvus_client
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.naive_rag.generator import BamlGenerationClient, generate_answer
from legal_rag.naive_rag.retriever import RetrievedChunk, Retriever, SearchMode
from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Container for a pipeline response."""

    query: str
    answer: str
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
        logger.info("NaiveRAGPipeline initialized")

    def run(self, query: str, top_k: int | None = None, search_mode: str | None = None) -> RAGResult:
        """Retrieve chunks and generate an answer, optionally overriding top_k and search mode."""
        k = top_k or self.settings.top_k
        mode = SearchMode(search_mode) if search_mode else None
        hits = self._retriever.retrieve(query, top_k=k, mode=mode)
        answer = generate_answer(query, hits, self._generation_client)
        active = search_mode or self.settings.search_mode
        logger.info("Pipeline completed: {} sources, {} search", len(hits), active)
        return RAGResult(query=query, answer=answer, sources=tuple(hits))
