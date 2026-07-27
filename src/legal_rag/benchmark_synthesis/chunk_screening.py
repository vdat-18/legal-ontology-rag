"""Filter low-signal chunks before spending fact-extraction LLM calls on them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ChunkScreeningResult
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings


@dataclass(frozen=True, slots=True)
class ExcludedChunk:
    """One chunk excluded from fact extraction, with an auditable reason."""

    chunk_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable exclusion record."""
        return {"chunk_id": self.chunk_id, "reason": self.reason}


class ChunkScreeningClient(Protocol):
    """Boundary implemented by a chunk-screening language-model client."""

    def screen(self, chunk_text: str) -> ChunkScreeningResult:
        """Return whether this chunk contains penalty content worth extracting."""


class BamlChunkScreeningClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, evaluation_settings: EvaluationSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=evaluation_settings.chunk_screening_max_output_tokens,
            temperature=evaluation_settings.chunk_screening_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = evaluation_settings.max_response_bytes

    def screen(self, chunk_text: str) -> ChunkScreeningResult:
        """Run one bounded BAML call and return its generated typed result."""
        request = b.request.ScreenChunk(chunk_text, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ScreenChunk(raw_output, baml_options={"client_registry": self._registry})


def screen_chunks(
    chunks: Sequence[Chunk],
    client: ChunkScreeningClient,
) -> tuple[tuple[Chunk, ...], tuple[ExcludedChunk, ...]]:
    """Split chunks into those worth reading for fact extraction and those excluded."""
    accepted: list[Chunk] = []
    excluded: list[ExcludedChunk] = []
    for chunk in chunks:
        result = client.screen(chunk.text)
        if result.has_penalty_content:
            accepted.append(chunk)
        else:
            excluded.append(ExcludedChunk(chunk_id=chunk.chunk_id, reason=result.reason))
    return tuple(accepted), tuple(excluded)
