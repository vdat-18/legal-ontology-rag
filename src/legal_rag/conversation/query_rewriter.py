"""Multi-turn query rewriter backed by BAML LLM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ConversationTurn as BamlConversationTurn
from legal_rag.baml_client.types import MultiTurnRewrittenQuery
from legal_rag.baml_transport import BamlCaller
from legal_rag.settings import Settings


class QueryRewriteClient(Protocol):
    """Boundary implemented by a query-rewriting language-model client."""

    def rewrite(self, query: str, history: Sequence[BamlConversationTurn] = ()) -> str:
        """Return *query* as a standalone question in formal legal terminology."""


class BamlMultiTurnQueryRewriter:
    """Typed BAML client for multi-turn query rewriting."""

    def __init__(
        self,
        settings: Settings,
        temperature: float = 0.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        self._caller = BamlCaller.from_settings(
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
        request = b.request.RewriteMultiTurnQuery(query, list(history), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        result: MultiTurnRewrittenQuery = b.parse.RewriteMultiTurnQuery(
            raw_output, baml_options=self._caller.options
        )
        return result.query_vi
