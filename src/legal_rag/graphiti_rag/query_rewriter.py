"""Rewrite colloquial legal questions into formal terminology before graph search.

Retrieval quality on the Graphiti graph is sensitive to vocabulary: the graph stores facts
using the formal terms legal documents actually use (e.g. "không chấp hành hiệu lệnh của đèn
tín hiệu giao thông"), while users naturally ask in colloquial terms (e.g. "vượt đèn đỏ").
A colloquial query can fail to retrieve the correct fact even though it exists in the graph.
Rewriting the query into formal terminology before searching closes most of this gap.

DEPRECATED: Use legal_rag.query_rewriting.AsyncQueryRewriter instead.
This module provides backward compatibility.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol

from legal_rag.baml_client.types import ConversationTurn as BamlConversationTurn
from legal_rag.query_rewriting import AsyncQueryRewriter
from legal_rag.settings import GraphitiSettings, Settings


class QueryRewriteClient(Protocol):
    """Boundary implemented by a query-rewriting language-model client."""

    async def rewrite(self, query: str, history: Sequence[BamlConversationTurn] = ()) -> str:
        """Return *query* as a standalone question in formal legal terminology."""


class BamlQueryRewriteClient:
    """Typed BAML client, backed by the shared runtime OpenAI-compatible provider.

    DEPRECATED: Use legal_rag.query_rewriting.AsyncQueryRewriter instead.
    """

    def __init__(self, settings: Settings, graphiti_settings: GraphitiSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration.

        Args:
            settings: Application LLM connection settings.
            graphiti_settings: Graphiti-specific settings for output token limits.
        """
        warnings.warn(
            "BamlQueryRewriteClient is deprecated. Use legal_rag.query_rewriting.AsyncQueryRewriter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._impl = AsyncQueryRewriter(
            settings,
            max_output_tokens=graphiti_settings.query_rewrite_max_output_tokens,
            temperature=graphiti_settings.extraction_temperature,
            max_response_bytes=graphiti_settings.max_response_bytes,
        )

    async def rewrite(self, query: str, history: Sequence[BamlConversationTurn] = ()) -> str:
        """Resolve the question against earlier turns and put it in formal legal terms.

        Both jobs run in one call: a follow-up like "còn xe máy thì sao?" is meaningless to the
        graph until the earlier subject is restored, and restoring it separately would cost a
        second round trip on every turn.
        """
        tuple_history = tuple((turn.question, turn.answer) for turn in history)
        result = await self._impl.rewrite(query, tuple_history)
        return result.rewritten
