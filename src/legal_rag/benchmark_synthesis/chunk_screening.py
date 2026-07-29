"""Filter low-signal chunks before spending fact-extraction LLM calls on them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings

DEFAULT_SCREENING_BATCH_SIZE = 10

CHUNK_SCREENING_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ChunkScreeningRecord:
    """One chunk's screening verdict, persisted for the next pipeline stage to read."""

    chunk_id: str
    has_penalty_content: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable screening record."""
        return {
            "schema_version": CHUNK_SCREENING_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "has_penalty_content": self.has_penalty_content,
            "reason": self.reason,
        }


class ChunkScreeningClient(Protocol):
    """Boundary implemented by a chunk-screening language-model client."""

    def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
        """Return, for each given chunk, whether it contains penalty content worth extracting."""


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

    def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
        """Run one bounded BAML call over a batch of chunks and return their typed results."""
        request = b.request.ScreenChunks(list(chunks), baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ScreenChunks(raw_output, baml_options={"client_registry": self._registry})


def screen_chunks(
    chunks: Sequence[Chunk],
    client: ChunkScreeningClient,
    *,
    batch_size: int = DEFAULT_SCREENING_BATCH_SIZE,
) -> tuple[ChunkScreeningRecord, ...]:
    """Screen every chunk and return one verdict record per chunk, in input order."""
    results_by_chunk_id: dict[str, ChunkScreeningResult] = {}
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        items = [ChunkScreeningItem(chunk_id=chunk.chunk_id, text=chunk.text) for chunk in batch]
        for result in client.screen(items):
            results_by_chunk_id[result.chunk_id] = result
        logger.debug("Screened {}/{} chunks", len(results_by_chunk_id), len(chunks))

    return tuple(
        ChunkScreeningRecord(
            chunk_id=chunk.chunk_id,
            has_penalty_content=results_by_chunk_id[chunk.chunk_id].has_penalty_content,
            reason=results_by_chunk_id[chunk.chunk_id].reason,
        )
        for chunk in chunks
    )
