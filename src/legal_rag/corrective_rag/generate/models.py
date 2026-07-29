"""Shared models for the Corrective RAG pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

KnowledgeAction = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]
KnowledgeKind = Literal["internal", "external"]


@dataclass(frozen=True, slots=True)
class GradedChunk:
    """A retrieved chunk with evaluator confidence."""

    chunk: Mapping[str, Any]
    score: float
    action: KnowledgeAction
    rationale: str = ""

    @property
    def is_relevant(self) -> bool:
        """Return whether the chunk is strong enough for internal knowledge."""
        return self.action == "CORRECT"


@dataclass(frozen=True, slots=True)
class KnowledgeItem:
    """Refined knowledge passed to the answer generator."""

    text: str
    source_name: str
    score: float
    kind: KnowledgeKind
    chunk_id: str | None = None
    url: str | None = None
