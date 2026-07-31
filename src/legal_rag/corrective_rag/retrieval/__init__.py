"""Retrieval adapters for Corrective RAG."""

from __future__ import annotations

from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel

__all__ = ["EmbeddingModel", "LegalVectorDB"]
