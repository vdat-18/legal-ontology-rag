"""Synchronous query rewriter using BAML LLM client."""

from __future__ import annotations

from collections.abc import Sequence

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ConversationTurn as BamlConversationTurn
from legal_rag.baml_client.types import RewrittenQuery as BamlRewrittenQuery
from legal_rag.baml_transport import BamlCaller
from legal_rag.query_rewriting.base import RewrittenQuery
from legal_rag.settings import Settings


class SyncQueryRewriter:
    """Synchronous query rewriter for blocking contexts.

    Resolves context-dependent follow-up questions and normalizes Vietnamese legal terminology.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> None:
        """Initialize the synchronous query rewriter.

        Args:
            settings: Application LLM connection settings.
            max_output_tokens: Maximum tokens for LLM response.
            temperature: LLM temperature for response generation.
            max_response_bytes: Maximum response size in bytes.
        """
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            max_response_bytes=max_response_bytes,
        )

    def rewrite(self, query: str, history: Sequence[tuple[str, str]] | tuple[tuple[str, str], ...] = ()) -> RewrittenQuery:
        """Return query as standalone question in formal legal terminology.

        Args:
            query: The user's raw question, possibly context-dependent.
            history: Sequence of (question, answer) pairs from earlier turns.

        Returns:
            RewrittenQuery with both original and rewritten query.
        """
        baml_history = [BamlConversationTurn(question=q, answer=a) for q, a in history]
        request = b.request.RewriteQueryToLegalTerms(query, baml_history, baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        result: BamlRewrittenQuery = b.parse.RewriteQueryToLegalTerms(raw_output, baml_options=self._caller.options)
        return RewrittenQuery(original=query, rewritten=result.query_vi)
