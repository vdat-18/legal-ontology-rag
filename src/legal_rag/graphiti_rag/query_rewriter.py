"""Rewrite colloquial legal questions into formal terminology before graph search.

Retrieval quality on the Graphiti graph is sensitive to vocabulary: the graph stores facts
using the formal terms legal documents actually use (e.g. "không chấp hành hiệu lệnh của đèn
tín hiệu giao thông"), while users naturally ask in colloquial terms (e.g. "vượt đèn đỏ").
A colloquial query can fail to retrieve the correct fact even though it exists in the graph.
Rewriting the query into formal terminology before searching closes most of this gap.
"""

from __future__ import annotations

from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import BamlCaller
from legal_rag.settings import GraphitiSettings, Settings


class QueryRewriteClient(Protocol):
    """Boundary implemented by a query-rewriting language-model client."""

    async def rewrite(self, query: str) -> str:
        """Return *query* rewritten into formal legal terminology."""


class BamlQueryRewriteClient:
    """Typed BAML client, backed by the shared runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, graphiti_settings: GraphitiSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=graphiti_settings.query_rewrite_max_output_tokens,
            temperature=graphiti_settings.extraction_temperature,
            max_response_bytes=graphiti_settings.max_response_bytes,
        )

    async def rewrite(self, query: str) -> str:
        """Run one bounded BAML call and return the query rewritten into formal legal terms."""
        request = b.request.RewriteQueryToLegalTerms(query, baml_options=self._caller.options)
        raw_output = await self._caller.execute_async(request)
        return b.parse.RewriteQueryToLegalTerms(raw_output, baml_options=self._caller.options).query_vi
