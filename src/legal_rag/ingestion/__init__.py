"""Ingestion: turn source documents into time-stamped Graphiti episodes."""

from __future__ import annotations

from legal_rag.ingestion.documents import LegalEpisode, iter_episodes
from legal_rag.ingestion.raw import RawDocument, RawMeta, iter_mock_raw_documents, mock_raw_dir

__all__ = [
    "LegalEpisode",
    "RawDocument",
    "RawMeta",
    "iter_episodes",
    "iter_mock_raw_documents",
    "mock_raw_dir",
]
