"""Flag legal document citations that reference documents outside the indexed corpus.

Graphiti can correctly extract a real cross-reference from a document's own text (e.g. "168/2024/
NĐ-CP sửa đổi một số điều của Nghị định số 100/2019/NĐ-CP") into a graph fact or node — this is
not a hallucination, but it does mean the *referenced* document (100/2019/NĐ-CP here) may not
itself be indexed, so any of its un-amended provisions are invisible to this graph. Surfacing
this distinction lets an answer be labeled "may be incomplete" rather than silently presented
as fully grounded.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import BamlCaller
from legal_rag.settings import GraphitiSettings, Settings


class DocumentCitationClient(Protocol):
    """Boundary implemented by a document-citation-extracting language-model client."""

    def extract_citations(self, text: str) -> list[str]:
        """Return the document numbers cited in *text*."""


class BamlDocumentCitationClient:
    """Typed BAML client, backed by the shared runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, graphiti_settings: GraphitiSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=graphiti_settings.citation_max_output_tokens,
            temperature=graphiti_settings.extraction_temperature,
            max_response_bytes=graphiti_settings.max_response_bytes,
        )

    def extract_citations(self, text: str) -> list[str]:
        """Run one bounded BAML call and return the document numbers cited in *text*."""
        request = b.request.ExtractCitedDocuments(text, baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        result = b.parse.ExtractCitedDocuments(raw_output, baml_options=self._caller.options)
        return list(result.document_numbers)


def _appears_as_complete_span(text: str, candidate: str) -> bool:
    """Return whether *candidate* occurs outside a longer legal-document-number token."""
    if not candidate:
        return False

    start = text.find(candidate)
    while start >= 0:
        end = start + len(candidate)
        left_is_token = start > 0 and (text[start - 1].isalnum() or text[start - 1] in "/-")
        right_is_token = end < len(text) and (text[end].isalnum() or text[end] in "/-")
        if not left_is_token and not right_is_token:
            return True
        start = text.find(candidate, start + 1)
    return False


def _document_number_identity(document_number: str) -> str:
    """Return an accent-insensitive identity key for one Vietnamese document number."""
    decomposed = unicodedata.normalize("NFKD", document_number)
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return "".join(without_marks.replace("Đ", "D").replace("đ", "d").upper().split())


def find_unindexed_document_citations(
    texts: Sequence[str], known_document_numbers: frozenset[str], client: DocumentCitationClient
) -> list[str]:
    """Return legal document numbers cited in *texts* that aren't in *known_document_numbers*.

    Args:
        texts: Fact or answer text to scan for cited legal documents.
        known_document_numbers: Document numbers actually indexed into this corpus.
        client: The language-model client that identifies cited documents.

    Returns:
        Sorted, deduplicated document numbers cited in *texts* but absent from *known_document_numbers*.
        A returned number the model did not actually copy from *texts* is discarded, since this
        result is surfaced to the reader as a caveat about the answer's coverage.
    """
    scanned = "\n".join(texts)
    cited = sorted(
        {number for number in client.extract_citations(scanned) if _appears_as_complete_span(scanned, number)}
    )
    known_identities = {_document_number_identity(number) for number in known_document_numbers}
    unindexed_by_identity: dict[str, str] = {}
    for number in cited:
        identity = _document_number_identity(number)
        if identity not in known_identities:
            unindexed_by_identity.setdefault(identity, number)
    return sorted(unindexed_by_identity.values())
