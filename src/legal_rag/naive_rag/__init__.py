"""Naive RAG baseline (B0) for Vietnamese traffic law QA."""

from legal_rag.naive_rag.config import NaiveRAGConfig
from legal_rag.naive_rag.pipeline import NaiveRAGPipeline, RAGResult

__all__ = ["NaiveRAGConfig", "NaiveRAGPipeline", "RAGResult"]
