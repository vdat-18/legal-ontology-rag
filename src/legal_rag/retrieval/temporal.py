"""Point-in-time (bi-temporal) retrieval over the penalty graph.

Runs Graphiti's hybrid search, then keeps only facts whose bi-temporal
``valid_at`` / ``invalid_at`` window contains the reference moment — so the same
question answered at two dates returns the provision valid at each date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from graphiti_core.edges import EntityEdge

from legal_rag.config import Settings
from legal_rag.graph.client import build_graphiti


@dataclass(frozen=True)
class TemporalFact:
    """A fact together with its validity window."""

    fact: str
    valid_from: datetime | None
    valid_to: datetime | None


def _valid_at(edge: EntityEdge, moment: datetime) -> bool:
    """True if the edge's fact holds at ``moment`` (bi-temporal window)."""
    if edge.valid_at is not None and edge.valid_at > moment:
        return False
    return not (edge.invalid_at is not None and edge.invalid_at <= moment)


async def query_at(settings: Settings, query: str, moment: datetime, limit: int = 20) -> list[TemporalFact]:
    """Return facts matching ``query`` that are valid at ``moment``."""
    graphiti = build_graphiti(settings)
    try:
        edges = await graphiti.search(query, group_ids=[settings.group_id], num_results=limit)
        return [
            TemporalFact(fact=e.fact, valid_from=e.valid_at, valid_to=e.invalid_at)
            for e in edges
            if _valid_at(e, moment)
        ]
    finally:
        await graphiti.close()
