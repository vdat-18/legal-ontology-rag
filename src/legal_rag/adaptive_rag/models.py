"""Data models for adaptive RAG planning and results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from legal_rag.naive_rag.models import RetrievedChunk


class AdaptiveRoute(Enum):
    """Supported one-shot retrieval routes."""

    DIRECT = "direct"
    MULTI_HOP = "multi_hop"


@dataclass(frozen=True, slots=True)
class AdaptiveQueryPlan:
    """Validated route and standalone retrieval subqueries."""

    route: AdaptiveRoute
    subqueries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdaptiveRAGResult:
    """Answer, retrieval provenance, and the planner decision used to produce it."""

    query: str
    rewritten_query: str | None = None
    answer: str = ""
    sources: tuple[RetrievedChunk, ...] = ()
    route: AdaptiveRoute = AdaptiveRoute.DIRECT
    subqueries: tuple[str, ...] = ()
