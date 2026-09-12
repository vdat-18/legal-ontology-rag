"""Multi-turn query rewriter backed by BAML LLM.

DEPRECATED: Use legal_rag.query_rewriting.SyncQueryRewriter instead.
This module provides backward compatibility.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol

from legal_rag.baml_client.types import ConversationTurn as BamlConversationTurn
from legal_rag.query_rewriting import SyncQueryRewriter
from legal_rag.settings import Settings


class QueryRewriteClient(Protocol):
    """Boundary implemented by a query-rewriting language-model client."""

    def rewrite(self, query: str, history: Sequence[BamlConversationTurn] = ()) -> str:
        """Return *query* as a standalone question in formal legal terminology."""


class BamlMultiTurnQueryRewriter:
    """Typed BAML client for multi-turn query rewriting.

    DEPRECATED: Use legal_rag.query_rewriting.SyncQueryRewriter instead.
    """

    def __init__(
        self,
        settings: Settings,
        temperature: float = 0.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration.

        Args:
            settings: Application LLM connection settings.
            temperature: LLM temperature for response generation.
            max_response_bytes: Maximum response size in bytes.
        """
        warnings.warn(
            "BamlMultiTurnQueryRewriter is deprecated. Use legal_rag.query_rewriting.SyncQueryRewriter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._impl = SyncQueryRewriter(
            settings,
            max_output_tokens=4096,
            temperature=temperature,
            max_response_bytes=max_response_bytes,
        )

    def rewrite(self, query: str, history: Sequence[BamlConversationTurn] = ()) -> str:
        """Resolve the question against earlier turns and put it in formal legal terms.

        Both jobs run in one call: a follow-up like "còn xe máy thì sao?" is meaningless to the
        retriever until the earlier subject is restored, and restoring it separately would cost a
        second round trip on every turn.
        """
        tuple_history = tuple((turn.question, turn.answer) for turn in history)
        result = self._impl.rewrite(query, tuple_history)
        return result.rewritten
