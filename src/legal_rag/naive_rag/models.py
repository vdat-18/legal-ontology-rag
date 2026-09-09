"""Data models for Naive RAG."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """Supported retrieval strategies."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float
