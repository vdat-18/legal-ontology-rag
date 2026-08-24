"""Filter low-signal chunks before spending fact-extraction LLM calls on them."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.llm_execution import (
    IncompleteLlmTasksError,
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.chunker.models import Chunk
from legal_rag.settings import Settings

CHUNK_SCREENING_SCHEMA_VERSION = "2.0.0"
CHUNK_SCREENING_PROMPT_VERSION = "5.1.0"


@dataclass(frozen=True, slots=True)
class ChunkScreeningRecord:
    """One chunk's screening verdict, persisted for the next pipeline stage to read."""

    chunk_id: str
    has_penalty_content: bool
    reason: str
    has_document_metadata: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable screening record."""
        return {
            "schema_version": CHUNK_SCREENING_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "has_penalty_content": self.has_penalty_content,
            "has_document_metadata": self.has_document_metadata,
            "reason": self.reason,
        }


class ChunkScreeningClient(Protocol):
    """Boundary implemented by a chunk-screening language-model client."""

    def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
        """Return, for each given chunk, whether it contains penalty content worth extracting."""


class BamlChunkScreeningClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.chunk_screening.max_output_tokens,
            temperature=synthesis_settings.chunk_screening.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.chunk_screening.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

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
            has_document_metadata=results_by_chunk_id[chunk.chunk_id].has_document_metadata,
            reason=results_by_chunk_id[chunk.chunk_id].reason,
        )
        for chunk in chunks
        if chunk.chunk_id in results_by_chunk_id
    )


def _unique_requested_results(
    items: Sequence[ChunkScreeningItem],
    results: Sequence[ChunkScreeningResult],
) -> tuple[dict[str, ChunkScreeningResult], int, int]:
    """Return uniquely matched verdicts plus duplicate and unexpected ID counts."""
    requested_ids = {item.chunk_id for item in items}
    result_counts = Counter(result.chunk_id for result in results)
    matching = {
        result.chunk_id: result
        for result in results
        if result.chunk_id in requested_ids and result_counts[result.chunk_id] == 1
    }
    duplicate_id_count = sum(count > 1 for chunk_id, count in result_counts.items() if chunk_id in requested_ids)
    unexpected_id_count = sum(chunk_id not in requested_ids for chunk_id in result_counts)
    return matching, duplicate_id_count, unexpected_id_count


def _screen_batch_with_checkpoints(
    batch_number: int,
    batch_count: int,
    total_chunk_count: int,
    batch: Sequence[Chunk],
    items: Sequence[ChunkScreeningItem],
    task_key: str,
    client: ChunkScreeningClient,
    results_by_chunk_id: dict[str, ChunkScreeningResult],
    checkpoint_lock: Lock,
    on_batch_screened: Callable[[Sequence[ChunkScreeningRecord]], None] | None,
    max_retries: int,
    retry_delay_seconds: float,
) -> None:
    """Checkpoint valid verdicts and retry only unresolved items in one batch."""
    logger.info(
        "Chunk screening batch {}/{} started: chunks={}",
        batch_number,
        batch_count,
        len(items),
    )
    pending_items = {item.chunk_id: item for item in items}

    def _screen_pending() -> None:
        pending = tuple(pending_items.values())
        results = client.screen(pending)
        matching, duplicate_id_count, unexpected_id_count = _unique_requested_results(pending, results)
        if matching:
            new_records = _records_for(batch, matching)
            with checkpoint_lock:
                results_by_chunk_id.update(matching)
                if on_batch_screened is not None:
                    on_batch_screened(new_records)
                progress = len(results_by_chunk_id)
            for chunk_id in matching:
                del pending_items[chunk_id]
            logger.info(
                "Chunk screening batch {}/{} checkpointed: completed={}, remaining={}, progress={}/{} chunks",
                batch_number,
                batch_count,
                len(matching),
                len(pending_items),
                progress,
                total_chunk_count,
            )
        if pending_items:
            msg = (
                f"Chunk screening response left {len(pending_items)} chunk(s) incomplete; "
                f"duplicate_ids={duplicate_id_count}, unexpected_ids={unexpected_id_count}"
            )
            raise ValueError(msg)

    execute_llm_task(
        task_key,
        _screen_pending,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    with checkpoint_lock:
        batch_results = tuple(results_by_chunk_id[chunk.chunk_id] for chunk in batch)
        progress = len(results_by_chunk_id)
    accepted_count = sum(result.has_penalty_content for result in batch_results)
    logger.info(
        "Chunk screening batch {}/{} completed: accepted={}, excluded={}, progress={}/{} chunks",
        batch_number,
        batch_count,
        accepted_count,
        len(batch_results) - accepted_count,
        progress,
        total_chunk_count,
    )


def screen_chunks(
    chunks: Sequence[Chunk],
    client: ChunkScreeningClient,
    *,
    batch_size: int,
    max_concurrency: int = 1,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_screened: Callable[[Sequence[ChunkScreeningRecord]], None] | None = None,
) -> tuple[ChunkScreeningRecord, ...]:
    """Screen chunks in bounded concurrent batches and checkpoint each successful batch."""
    results_by_chunk_id: dict[str, ChunkScreeningResult] = {}
    failures: list[LlmTaskFailure] = []
    checkpoint_lock = Lock()
    batches = [chunks[batch_start : batch_start + batch_size] for batch_start in range(0, len(chunks), batch_size)]
    batch_count = len(batches)
    logger.info(
        "Chunk screening started: {} chunks, {} batches, batch_size={}, max_concurrency={}",
        len(chunks),
        batch_count,
        batch_size,
        max_concurrency,
    )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {}
        for batch_index, batch in enumerate(batches):
            items = [
                ChunkScreeningItem(
                    chunk_id=chunk.chunk_id,
                    context_text=chunk.context_text,
                    text=chunk.text,
                )
                for chunk in batch
            ]
            task_key = f"chunks:{batch[0].chunk_id}:{batch[-1].chunk_id}"
            future = executor.submit(
                _screen_batch_with_checkpoints,
                batch_index + 1,
                batch_count,
                len(chunks),
                batch,
                items,
                task_key,
                client,
                results_by_chunk_id,
                checkpoint_lock,
                on_batch_screened,
                max_retries,
                retry_delay_seconds,
            )
            futures[future] = (batch_index + 1, task_key)

        for future in as_completed(futures):
            batch_number, task_key = futures[future]
            try:
                future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=task_key, error=str(error)))
                logger.error("Chunk screening batch {}/{} failed", batch_number, batch_count)

    with checkpoint_lock:
        records = _records_for(chunks, results_by_chunk_id)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("chunk screening", failures)
        raise incomplete_error
    return records
