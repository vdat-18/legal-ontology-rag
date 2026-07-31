"""Build reading windows and extract legal facts from chunk text."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
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

EXTRACTION_PROGRESS_SCHEMA_VERSION = "1.0.0"
FACT_EXTRACTION_PROMPT_VERSION = "1.2.0"


@dataclass(frozen=True, slots=True)
class ExtractionProgressRecord:
    """One chunk fact extraction has finished reading, recorded whether or not it produced facts."""

    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable extraction-progress record."""
        return {
            "schema_version": EXTRACTION_PROGRESS_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
        }


def _walk_adjacent_chunks(
    start_chunk_id: str | None,
    chunks_by_id: Mapping[str, Chunk],
    entries: dict[str, ChunkWindowEntry],
    *,
    max_chunks: int,
    next_chunk_id: Callable[[Chunk], str | None],
) -> None:
    """Walk a previous/next chunk chain, adding each hop found to entries as Adjacent context."""
    chunk_id = start_chunk_id
    for _ in range(max_chunks):
        if chunk_id is None or chunk_id in entries:
            return
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            return
        entries[chunk_id] = ChunkWindowEntry(chunk_id=chunk_id, role=ChunkWindowRole.Adjacent, text=chunk.text)
        chunk_id = next_chunk_id(chunk)


def build_reading_window(
    target: Chunk,
    chunks_by_id: Mapping[str, Chunk],
    *,
    max_preceding_chunks: int,
    max_following_chunks: int,
) -> tuple[ChunkWindowEntry, ...]:
    """Assemble one reading window around a target chunk."""
    entries: dict[str, ChunkWindowEntry] = {
        target.chunk_id: ChunkWindowEntry(chunk_id=target.chunk_id, role=ChunkWindowRole.Target, text=target.text)
    }
    _walk_adjacent_chunks(
        target.previous_chunk_id,
        chunks_by_id,
        entries,
        max_chunks=max_preceding_chunks,
        next_chunk_id=lambda chunk: chunk.previous_chunk_id,
    )
    _walk_adjacent_chunks(
        target.next_chunk_id,
        chunks_by_id,
        entries,
        max_chunks=max_following_chunks,
        next_chunk_id=lambda chunk: chunk.next_chunk_id,
    )

    return tuple(entries.values())


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
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
        """Run one bounded BAML call over a batch of reading windows."""
        request = b.request.ExtractLegalFactBatch(list(items), baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ExtractLegalFactBatch(raw_output, baml_options={"client_registry": self._registry})


def build_extracted_fact(raw: BamlLegalFact, source_chunk_id: str, metadata: DocumentMetadataRecord) -> LegalFact:
    """Assign identity to one LLM-produced fact, stamping document metadata the window cannot carry."""
    return build_legal_fact(
        conduct=raw.conduct,
        subject=raw.subject,
        conditions=tuple(raw.conditions),
        monetary_penalty=raw.monetary_penalty,
        license_points_deducted=raw.license_points_deducted,
        additional_penalties=tuple(raw.additional_penalties),
        remedial_measures=tuple(raw.remedial_measures),
        valid_from=date.fromisoformat(raw.valid_from) if raw.valid_from else metadata.valid_from,
        valid_to=date.fromisoformat(raw.valid_to) if raw.valid_to else metadata.valid_to,
        document_number=metadata.document_number,
        provision_path=raw.provision_path,
        evidence=tuple(
            EvidenceReference(chunk_id=entry.chunk_id, quoted_span=entry.quoted_span) for entry in raw.evidence
        ),
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
    return tuple(
        (
            target_chunk_id,
            tuple(
                build_extracted_fact(raw_fact, target_chunk_id, metadata)
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
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[LegalFact, ...]]]], None] | None = None,
) -> tuple[LegalFact, ...]:
    """Extract legal facts in batches, persisting each successful batch before continuing."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in document_chunks}

    facts: list[LegalFact] = []
    failures: list[LlmTaskFailure] = []
    for batch_start in range(0, len(target_chunks), batch_size):
        batch = target_chunks[batch_start : batch_start + batch_size]
        items = tuple(
            FactExtractionBatchItem(
                target_chunk_id=target.chunk_id,
                window=list(
                    build_reading_window(
                        target,
                        chunks_by_id,
                        max_preceding_chunks=max_preceding_chunks,
                        max_following_chunks=max_following_chunks,
                    )
                ),
            )
            for target in batch
        )
        task_key = _batch_task_key(items)

        try:
            batch_results = execute_llm_task(
                task_key,
                partial(_extract_batch_facts, client, items, metadata),
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        except LlmTaskExhaustedError as error:
            failures.append(LlmTaskFailure(task_key=task_key, error=str(error)))
        else:
            for _, target_facts in batch_results:
                facts.extend(target_facts)
            if on_batch_processed is not None:
                on_batch_processed(batch_results)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("fact extraction", failures)
        raise incomplete_error
    return tuple(facts)
