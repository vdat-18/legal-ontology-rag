"""Build reading windows and extract legal facts from chunk text."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable, Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ChunkWindowEntry,
    ChunkWindowRole,
    FactExtractionBatchItem,
    FactExtractionBatchResult,
)
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.llm_execution import (
    IncompleteLlmTasksError,
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact, build_legal_fact
from legal_rag.chunker.models import Chunk
from legal_rag.settings import Settings

LEGACY_EXTRACTION_PROGRESS_SCHEMA_VERSION = "1.0.0"
EXTRACTION_PROGRESS_SCHEMA_VERSION = "2.0.0"
FACT_EXTRACTION_PROMPT_VERSION = "5.6.0"


@dataclass(frozen=True, slots=True)
class ExtractionProgressRecord:
    """One chunk fact extraction has finished reading, recorded whether or not it produced facts."""

    chunk_id: str
    prompt_version: str | None = None
    model: str | None = None
    extraction_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable extraction-progress record."""
        return {
            "schema_version": EXTRACTION_PROGRESS_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "extraction_run_id": self.extraction_run_id,
        }


def _walk_adjacent_chunks(
    start_chunk_id: str | None,
    chunks_by_id: Mapping[str, Chunk],
    *,
    max_chunks: int,
    next_chunk_id: Callable[[Chunk], str | None],
) -> tuple[Chunk, ...]:
    """Return up to max_chunks neighbors by walking one linked direction."""
    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    chunk_id = start_chunk_id
    for _ in range(max_chunks):
        if chunk_id is None or chunk_id in seen_ids:
            break
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            break
        chunks.append(chunk)
        seen_ids.add(chunk_id)
        chunk_id = next_chunk_id(chunk)
    return tuple(chunks)


def _window_entry(
    chunk: Chunk,
    role: ChunkWindowRole,
    extraction_target_ids: Set[str],
) -> ChunkWindowEntry:
    """Return one BAML reading-window entry for a persisted chunk."""
    if chunk.chunk_id in extraction_target_ids:
        if role == ChunkWindowRole.Preceding:
            role = ChunkWindowRole.PrecedingTarget
        elif role == ChunkWindowRole.Following:
            role = ChunkWindowRole.FollowingTarget
    return ChunkWindowEntry(
        chunk_id=chunk.chunk_id,
        role=role,
        context_text=chunk.context_text,
        text=chunk.text,
    )


def _source_grounded_context_chunks(
    target: Chunk,
    chunks_by_id: Mapping[str, Chunk],
) -> tuple[Chunk, ...]:
    """Resolve parser ancestry to source chunks, ordered from most to least specific."""
    context_lines = tuple(dict.fromkeys(line.strip() for line in target.context_text.splitlines() if line.strip()))
    preceding_chunks = tuple(
        sorted(
            (
                chunk
                for chunk in chunks_by_id.values()
                if chunk.scope_id == target.scope_id and chunk.chunk_index < target.chunk_index
            ),
            key=lambda chunk: chunk.chunk_index,
        )
    )
    matched_by_id: dict[str, Chunk] = {}
    for line in reversed(context_lines):
        if line in target.text:
            continue
        source_chunk = next((chunk for chunk in reversed(preceding_chunks) if line in chunk.text), None)
        if source_chunk is not None:
            matched_by_id.setdefault(source_chunk.chunk_id, source_chunk)
    return tuple(matched_by_id.values())


def build_reading_window(
    target: Chunk,
    chunks_by_id: Mapping[str, Chunk],
    *,
    max_preceding_chunks: int,
    max_following_chunks: int,
    extraction_target_ids: Set[str] = frozenset(),
) -> tuple[ChunkWindowEntry, ...]:
    """Assemble local neighbors plus source chunks for parser-derived structural context."""
    preceding = _walk_adjacent_chunks(
        target.previous_chunk_id,
        chunks_by_id,
        max_chunks=max_preceding_chunks,
        next_chunk_id=lambda chunk: chunk.previous_chunk_id,
    )
    following = _walk_adjacent_chunks(
        target.next_chunk_id,
        chunks_by_id,
        max_chunks=max_following_chunks,
        next_chunk_id=lambda chunk: chunk.next_chunk_id,
    )
    local_window = (
        *(_window_entry(chunk, ChunkWindowRole.Preceding, extraction_target_ids) for chunk in reversed(preceding)),
        _window_entry(target, ChunkWindowRole.Target, extraction_target_ids),
        *(_window_entry(chunk, ChunkWindowRole.Following, extraction_target_ids) for chunk in following),
    )
    local_chunk_ids = {entry.chunk_id for entry in local_window}
    article_context = tuple(
        ChunkWindowEntry(
            chunk_id=chunk.chunk_id,
            role=ChunkWindowRole.ArticleContext,
            context_text=chunk.context_text,
            text=chunk.text,
        )
        for chunk in _source_grounded_context_chunks(target, chunks_by_id)
        if chunk.chunk_id not in local_chunk_ids
    )
    return article_context + local_window


class FactExtractionClient(Protocol):
    """Boundary implemented by a legal-fact-extraction language-model client."""

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
        """Return typed candidate facts for a batch of reading windows."""


class BamlFactExtractionClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.fact_extraction.max_output_tokens,
            temperature=synthesis_settings.fact_extraction.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.fact_extraction.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
        """Run one bounded BAML call over a batch of reading windows."""
        request = b.request.ExtractLegalFactBatch(list(items), baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ExtractLegalFactBatch(raw_output, baml_options={"client_registry": self._registry})


def build_extracted_fact(
    raw: BamlLegalFact,
    source_chunk_id: str,
    metadata: DocumentMetadataRecord,
    window_chunks_by_id: Mapping[str, None],
) -> LegalFact:
    """Attach chunk-level evidence and assign identity to one LLM-produced fact."""
    evidence = tuple(EvidenceReference(chunk_id=entry.chunk_id) for entry in raw.evidence)
    if not evidence:
        msg = "A legal fact must cite at least one evidence chunk"
        raise ValueError(msg)
    if source_chunk_id not in {reference.chunk_id for reference in evidence}:
        msg = "A legal fact must cite its target chunk"
        raise ValueError(msg)
    if any(
        character.isalpha() and not unicodedata.name(character, "").startswith("LATIN") for character in raw.statement
    ):
        msg = "A legal fact statement must use Latin-script Vietnamese text"
        raise ValueError(msg)
    for reference in evidence:
        window_chunks_by_id[reference.chunk_id]
    return build_legal_fact(
        statement=raw.statement,
        valid_from=metadata.valid_from,
        valid_to=metadata.valid_to,
        document_number=metadata.document_number,
        evidence=evidence,
        source_chunk_id=source_chunk_id,
    )


def _extract_batch_facts(
    client: FactExtractionClient,
    items: Sequence[FactExtractionBatchItem],
    metadata: DocumentMetadataRecord,
) -> tuple[tuple[str, tuple[LegalFact, ...]], ...]:
    """Extract one batch and reject missing, duplicate, or unexpected target results."""
    results = client.extract_batch(items)
    requested_ids = [item.target_chunk_id for item in items]
    result_ids = [result.target_chunk_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Fact extraction response must contain exactly one result for every requested target"
        raise ValueError(msg)
    results_by_target = {result.target_chunk_id: result for result in results}
    window_chunks_by_target = {item.target_chunk_id: {entry.chunk_id: None for entry in item.window} for item in items}
    return tuple(
        (
            target_chunk_id,
            tuple(
                build_extracted_fact(raw_fact, target_chunk_id, metadata, window_chunks_by_target[target_chunk_id])
                for raw_fact in results_by_target[target_chunk_id].facts
            ),
        )
        for target_chunk_id in requested_ids
    )


def _batch_task_key(items: Sequence[FactExtractionBatchItem]) -> str:
    """Build a deterministic retry key from the ordered target ids in one batch."""
    target_ids = "\0".join(item.target_chunk_id for item in items)
    return f"fact-batch:{hashlib.sha256(target_ids.encode('utf-8')).hexdigest()}"


def extract_facts_from_document(
    target_chunks: Sequence[Chunk],
    document_chunks: Sequence[Chunk],
    client: FactExtractionClient,
    metadata: DocumentMetadataRecord,
    *,
    max_preceding_chunks: int,
    max_following_chunks: int,
    batch_size: int,
    max_concurrency: int = 1,
    extraction_target_ids: Set[str] | None = None,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[LegalFact, ...]]]], None] | None = None,
) -> tuple[LegalFact, ...]:
    """Extract facts in bounded concurrent batches and checkpoint each successful batch."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in document_chunks}
    all_target_ids = extraction_target_ids or {chunk.chunk_id for chunk in target_chunks}

    batch_results_by_index: dict[int, tuple[tuple[str, tuple[LegalFact, ...]], ...]] = {}
    failures: list[LlmTaskFailure] = []
    batches = [
        target_chunks[batch_start : batch_start + batch_size]
        for batch_start in range(0, len(target_chunks), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {}
        for batch_index, batch in enumerate(batches):
            items = tuple(
                FactExtractionBatchItem(
                    target_chunk_id=target.chunk_id,
                    window=list(
                        build_reading_window(
                            target,
                            chunks_by_id,
                            max_preceding_chunks=max_preceding_chunks,
                            max_following_chunks=max_following_chunks,
                            extraction_target_ids=all_target_ids,
                        )
                    ),
                )
                for target in batch
            )
            task_key = _batch_task_key(items)
            future = executor.submit(
                execute_llm_task,
                task_key,
                partial(_extract_batch_facts, client, items, metadata),
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            futures[future] = (batch_index, task_key)

        for future in as_completed(futures):
            batch_index, task_key = futures[future]
            try:
                batch_results = future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=task_key, error=str(error)))
                continue
            batch_results_by_index[batch_index] = batch_results
            if on_batch_processed is not None:
                on_batch_processed(batch_results)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("fact extraction", failures)
        raise incomplete_error
    return tuple(
        fact
        for batch_index in range(len(batches))
        for _, target_facts in batch_results_by_index[batch_index]
        for fact in target_facts
    )
