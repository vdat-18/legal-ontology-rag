"""Flag legal document citations that reference documents outside the indexed corpus.

Graphiti can correctly extract a real cross-reference from a document's own text (e.g. "168/2024/
NĐ-CP sửa đổi một số điều của Nghị định số 100/2019/NĐ-CP") into a graph fact or node — this is
not a hallucination, but it does mean the *referenced* document (100/2019/NĐ-CP here) may not
itself be indexed, so any of its un-amended provisions are invisible to this graph. Surfacing
this distinction lets an answer be labeled "may be incomplete" rather than silently presented
as fully grounded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.settings import GraphitiSettings, Settings


class DocumentCitationClient(Protocol):
    """Boundary implemented by a document-citation-extracting language-model client."""

    def extract_citations(self, text: str) -> list[str]:
        """Return the document numbers cited in *text*."""


class BamlDocumentCitationClient:
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

    def extract_citations(self, text: str) -> list[str]:
        """Run one bounded BAML call and return the document numbers cited in *text*."""
        request = b.request.ExtractCitedDocuments(text, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        result = b.parse.ExtractCitedDocuments(raw_output, baml_options={"client_registry": self._registry})
        return list(result.document_numbers)


def find_unindexed_document_citations(
    texts: Sequence[str], known_so_hieu: frozenset[str], client: DocumentCitationClient
) -> list[str]:
    """Return legal document numbers cited in *texts* that aren't in *known_so_hieu*.

    Args:
        texts: Fact or answer text to scan for cited legal documents.
        known_so_hieu: The ``so_hieu`` values of documents actually indexed into this corpus.
        client: The language-model client that identifies cited documents.

    Returns:
        Sorted, deduplicated document numbers cited in *texts* but absent from *known_so_hieu*.
    """
    cited = set(client.extract_citations("\n".join(texts)))
    return sorted(cited - known_so_hieu)
