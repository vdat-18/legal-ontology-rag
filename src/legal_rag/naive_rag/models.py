"""Data models for Naive RAG."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """Supported retrieval strategies."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float
    unit_id: str | None = None
    unit_kind: str = "chunk"
    evidence_chunk_ids: tuple[str, ...] = ()
    content_sha256: str | None = None
    chunker_config_sha256: str | None = None

    @property
    def effective_unit_id(self) -> str:
        """Return the pipeline-native unit ID, falling back to the leaf chunk ID."""
        return self.unit_id or self.chunk_id

    @property
    def effective_evidence_chunk_ids(self) -> tuple[str, ...]:
        """Return source chunk evidence for evaluation and citations."""
        return self.evidence_chunk_ids or (self.chunk_id,)
