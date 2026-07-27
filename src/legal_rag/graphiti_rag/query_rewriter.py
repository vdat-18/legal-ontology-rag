"""Rewrite colloquial legal questions into formal terminology before graph search.

Retrieval quality on the Graphiti graph is sensitive to vocabulary: the graph stores facts
using the formal terms legal documents actually use (e.g. "không chấp hành hiệu lệnh của đèn
tín hiệu giao thông"), while users naturally ask in colloquial terms (e.g. "vượt đèn đỏ").
A colloquial query can fail to retrieve the correct fact even though it exists in the graph.
Rewriting the query into formal terminology before searching closes most of this gap.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.settings import GraphitiSettings, Settings


class QueryRewriteClient(Protocol):
    """Boundary implemented by a query-rewriting language-model client."""

    async def rewrite(self, query: str) -> str:
        """Return *query* rewritten into formal legal terminology."""


class BamlQueryRewriteClient:
    """Typed BAML client, backed by the shared runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, graphiti_settings: GraphitiSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=graphiti_settings.query_rewrite_max_output_tokens,
            temperature=0.0,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = graphiti_settings.max_response_bytes

    async def rewrite(self, query: str) -> str:
        """Run one bounded BAML call and return the query rewritten into formal legal terms."""
        request = b.request.RewriteQueryToLegalTerms(query, baml_options={"client_registry": self._registry})
        raw_output = await asyncio.to_thread(
            execute_baml_request, request, self._timeout_seconds, self._max_response_bytes
        )
        return b.parse.RewriteQueryToLegalTerms(raw_output, baml_options={"client_registry": self._registry}).query_vi
