"""Corrective RAG package.

Provides vector similarity search and relevance-verified answer generation using Milvus.
"""

from __future__ import annotations

from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.generate.generator import ResponseGenerator
from legal_rag.corrective_rag.generate.grader import CorrectiveGrader
from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline
from legal_rag.corrective_rag.generate.refiner import KnowledgeRefiner
from legal_rag.corrective_rag.generate.web_search import WebSearchProvider
from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel

__all__ = [
    "CorrectiveGrader",
    "CorrectiveRAGPipeline",
    "EmbeddingModel",
    "KnowledgeRefiner",
    "LegalVectorDB",
    "RAGConfig",
    "ResponseGenerator",
    "WebSearchProvider",
]
