"""Filter low-signal chunks before spending fact-extraction LLM calls on them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.llm_execution import (
    IncompleteLlmTasksError,
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings

DEFAULT_SCREENING_BATCH_SIZE = 20
DEFAULT_SCREENING_MAX_RETRIES = 2

CHUNK_SCREENING_SCHEMA_VERSION = "1.0.0"
CHUNK_SCREENING_PROMPT_VERSION = "1.0.0"


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


def _records_for(
    chunks: Sequence[Chunk], results_by_chunk_id: dict[str, ChunkScreeningResult]
) -> tuple[ChunkScreeningRecord, ...]:
    """Build a verdict record for each of chunks that already has a result."""
    return tuple(
        ChunkScreeningRecord(
            chunk_id=chunk.chunk_id,
            has_penalty_content=results_by_chunk_id[chunk.chunk_id].has_penalty_content,
            reason=results_by_chunk_id[chunk.chunk_id].reason,
        )
        for chunk in chunks
        if chunk.chunk_id in results_by_chunk_id
    )


def _screen_batch(
    client: ChunkScreeningClient,
    items: Sequence[ChunkScreeningItem],
) -> Sequence[ChunkScreeningResult]:
    """Screen one batch and reject missing, duplicate, or unexpected verdicts."""
    results = client.screen(items)
    requested_ids = [item.chunk_id for item in items]
    result_ids = [result.chunk_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Chunk screening response must contain exactly one verdict for every requested chunk"
        raise ValueError(msg)
    return results


def screen_chunks(
    chunks: Sequence[Chunk],
    client: ChunkScreeningClient,
    *,
    batch_size: int = DEFAULT_SCREENING_BATCH_SIZE,
    max_retries: int = DEFAULT_SCREENING_MAX_RETRIES,
    on_batch_screened: Callable[[Sequence[ChunkScreeningRecord]], None] | None = None,
) -> tuple[ChunkScreeningRecord, ...]:
    """Screen every chunk, retrying a failed batch before skipping it; call on_batch_screened after each batch."""
    results_by_chunk_id: dict[str, ChunkScreeningResult] = {}
    failures: list[LlmTaskFailure] = []
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        items = [ChunkScreeningItem(chunk_id=chunk.chunk_id, text=chunk.text) for chunk in batch]
        task_key = f"chunks:{batch[0].chunk_id}:{batch[-1].chunk_id}"
        try:
            results = execute_llm_task(
                task_key,
                lambda items=items: _screen_batch(client, items),
                max_retries=max_retries,
            )
        except LlmTaskExhaustedError as error:
            failures.append(LlmTaskFailure(task_key=task_key, error=str(error)))
        else:
            for result in results:
                results_by_chunk_id[result.chunk_id] = result
        logger.debug("Screened {}/{} chunks", len(results_by_chunk_id), len(chunks))
        if on_batch_screened is not None:
            on_batch_screened(_records_for(batch, results_by_chunk_id))

    records = _records_for(chunks, results_by_chunk_id)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("chunk screening", failures)
        raise incomplete_error
    return records
