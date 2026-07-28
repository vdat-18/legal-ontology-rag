"""Extract a legal document's own commencement date from its chunks, via BAML/LLM."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.chunker.models import Chunk
from legal_rag.settings import GraphitiSettings, Settings


class EffectiveDateClient(Protocol):
    """Boundary implemented by a commencement-date-extracting language-model client."""

    def extract(self, document_text: str) -> date | None:
        """Return the document's own commencement date found in *document_text*, if any."""


class BamlEffectiveDateClient:
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

    def extract(self, document_text: str) -> date | None:
        """Run one bounded BAML call and return the document's own commencement date, if stated."""
        request = b.request.ExtractEffectiveDate(document_text, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        result = b.parse.ExtractEffectiveDate(raw_output, baml_options={"client_registry": self._registry})
        return date.fromisoformat(result.effective_date) if result.effective_date else None


def _windows(ordered_chunks: Sequence[Chunk], window_token_budget: int) -> list[str]:
    """Group consecutive chunks into windows, each bounded by *window_token_budget* tokens.

    One BAML call over an entire long document becomes unreliable well before typical Vietnamese
    legal documents run out of length (confirmed empirically against this project's longest
    ground-truth document, ~77k tokens, which produced inconsistent answers across repeated
    calls at the same temperature). Splitting into token-bounded windows keeps every call within
    the range known to work; the budget is typed config (``GraphitiSettings``), and this bound is
    a batching concern, not a content-based rule — the LLM alone still decides whether a given
    window states the document's commencement date.
    """
    windows: list[str] = []
    current_chunks: list[Chunk] = []
    current_tokens = 0
    for chunk in ordered_chunks:
        if current_chunks and current_tokens + chunk.text_token_count > window_token_budget:
            windows.append("\n".join(c.text for c in current_chunks))
            current_chunks = []
            current_tokens = 0
        current_chunks.append(chunk)
        current_tokens += chunk.text_token_count
    if current_chunks:
        windows.append("\n".join(c.text for c in current_chunks))
    return windows


def extract_effective_date(
    document_chunks: Sequence[Chunk], client: EffectiveDateClient, window_token_budget: int
) -> date | None:
    """Find one document's own commencement date, scanning token-bounded windows from the end.

    Vietnamese legal documents state their own commencement date near the end (typically in an
    article titled "Hiệu lực thi hành"), so windows are checked last-to-first and the first
    non-null result is returned, keeping the common one-window case to a single call.
    """
    ordered = sorted(document_chunks, key=lambda chunk: chunk.chunk_index)
    for window_text in reversed(_windows(ordered, window_token_budget)):
        extracted = client.extract(window_text)
        if extracted is not None:
            return extracted
    return None
