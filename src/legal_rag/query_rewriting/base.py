"""Protocol and data structures for shared query rewriting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RewrittenQuery:
    """Result of query rewriting with both original and rewritten query."""

    original: str
    rewritten: str


class QueryRewriter(Protocol):
    """Protocol for query rewriting across RAG backends.

    Implementations resolve context-dependent follow-up questions into standalone
    queries and normalize colloquial Vietnamese legal terms into formal terminology.
    """

    def rewrite(self, query: str, history: tuple[tuple[str, str], ...] = ()) -> RewrittenQuery:
        """Return query as standalone question in formal legal terminology.

        Args:
            query: The user's raw question, possibly context-dependent.
            history: Tuple of (question, answer) pairs from earlier turns.

        Returns:
            RewrittenQuery with both original and rewritten query.
        """
