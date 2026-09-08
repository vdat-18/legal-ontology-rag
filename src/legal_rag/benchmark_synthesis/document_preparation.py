"""Persist screening and metadata artifacts for one document."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import replace
from pathlib import Path

from loguru import logger

from legal_rag.benchmark_synthesis.chunk_screening import (
    ChunkScreeningClient,
    ChunkScreeningRecord,
    screen_chunks,
)
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import (
    DocumentMetadataClient,
    DocumentMetadataRecord,
    resolve_document_metadata,
)
from legal_rag.benchmark_synthesis.writer import (
    read_chunk_screening,
    read_document_metadata,
    write_chunk_screening,
    write_document_metadata,
)
from legal_rag.chunker.models import Chunk


def prepare_chunk_screening(
    chunks: Sequence[Chunk],
    output_path: Path,
    client_factory: Callable[[], ChunkScreeningClient],
    settings: SynthesisSettings,
    *,
    rescreen_chunk_ids: Collection[str] = (),
    prompt_version: str | None = None,
    model: str | None = None,
    screening_run_id: str | None = None,
    existing_prompt_version: str | None = None,
    existing_model: str | None = None,
    existing_screening_run_id: str | None = None,
) -> tuple[ChunkScreeningRecord, ...]:
    """Resume chunk screening and persist each completed batch."""
    existing_records = read_chunk_screening(output_path) if output_path.exists() else ()
    existing_records = tuple(
        replace(
            record,
            prompt_version=record.prompt_version or existing_prompt_version,
            model=record.model or existing_model,
            screening_run_id=record.screening_run_id or existing_screening_run_id,
        )
        for record in existing_records
    )
    records_by_chunk_id = {record.chunk_id: record for record in existing_records}
    requested_ids = set(rescreen_chunk_ids)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    unknown_ids = requested_ids - chunk_ids
    if unknown_ids:
        msg = f"Unknown chunk ids requested for re-screening: {sorted(unknown_ids)}"
        raise ValueError(msg)
    if requested_ids:
        missing_unselected_ids = chunk_ids - requested_ids - records_by_chunk_id.keys()
        if missing_unselected_ids:
            msg = "Selective re-screening requires prior verdicts for every unselected chunk"
            raise ValueError(msg)
        for chunk_id in requested_ids:
            records_by_chunk_id.pop(chunk_id, None)
        logger.info("Chunk screening selectively invalidated {} chunks", len(requested_ids))
    chunks_to_screen = tuple(chunk for chunk in chunks if chunk.chunk_id not in records_by_chunk_id)
    if existing_records:
        logger.info("Chunk screening resumed: {}/{} chunks already completed", len(existing_records), len(chunks))

    def _persist_batch(batch_records: Sequence[ChunkScreeningRecord]) -> None:
        records_by_chunk_id.update({record.chunk_id: record for record in batch_records})
        write_chunk_screening(
            output_path,
            tuple(records_by_chunk_id[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in records_by_chunk_id),
        )

    if chunks_to_screen:
        screen_chunks(
            chunks_to_screen,
            client_factory(),
            context_chunks=chunks,
            batch_size=settings.chunk_screening_batch_size,
            max_concurrency=settings.chunk_screening_max_concurrency,
            max_retries=settings.llm_max_retries,
            retry_delay_seconds=settings.llm_retry_delay_seconds,
            on_batch_screened=_persist_batch,
            prompt_version=prompt_version,
            model=model,
            screening_run_id=screening_run_id,
        )

    records = tuple(records_by_chunk_id[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in records_by_chunk_id)
    write_chunk_screening(output_path, records)
    return records


def prepare_document_metadata(
    chunks: Sequence[Chunk],
    screening_records: Sequence[ChunkScreeningRecord],
    output_path: Path,
    client_factory: Callable[[], DocumentMetadataClient],
    settings: SynthesisSettings,
) -> DocumentMetadataRecord:
    """Reuse complete metadata or resolve and persist it once."""
    if output_path.exists():
        logger.info("Document metadata reused from {}", output_path)
        return read_document_metadata(output_path)
    record = resolve_document_metadata(
        chunks,
        screening_records,
        client_factory(),
        max_retries=settings.llm_max_retries,
        retry_delay_seconds=settings.llm_retry_delay_seconds,
    )
    write_document_metadata(output_path, record)
    return record


def prepare_document_artifacts(
    chunks: Sequence[Chunk],
    screening_output_path: Path,
    metadata_output_path: Path,
    screening_client_factory: Callable[[], ChunkScreeningClient],
    metadata_client_factory: Callable[[], DocumentMetadataClient],
    settings: SynthesisSettings,
    *,
    rescreen_chunk_ids: Collection[str] = (),
    prompt_version: str | None = None,
    model: str | None = None,
    screening_run_id: str | None = None,
    existing_prompt_version: str | None = None,
    existing_model: str | None = None,
    existing_screening_run_id: str | None = None,
    on_screening_complete: Callable[[Sequence[ChunkScreeningRecord]], None] | None = None,
) -> tuple[tuple[ChunkScreeningRecord, ...], DocumentMetadataRecord]:
    """Prepare screening before resolving document metadata."""
    source_name = chunks[0].source_name if chunks else "empty-document"
    logger.info("Step 1 started for {}: Step 1A runs before Step 1B", source_name)
    logger.info("Step 1A chunk screening started for {}", source_name)
    records = prepare_chunk_screening(
        chunks,
        screening_output_path,
        screening_client_factory,
        settings,
        rescreen_chunk_ids=rescreen_chunk_ids,
        prompt_version=prompt_version,
        model=model,
        screening_run_id=screening_run_id,
        existing_prompt_version=existing_prompt_version,
        existing_model=existing_model,
        existing_screening_run_id=existing_screening_run_id,
    )
    logger.info("Step 1A chunk screening completed for {}", source_name)
    if on_screening_complete is not None:
        on_screening_complete(records)
    logger.info("Step 1B document metadata started for {}", source_name)
    metadata = prepare_document_metadata(
        chunks,
        records,
        metadata_output_path,
        metadata_client_factory,
        settings,
    )
    logger.info("Step 1B document metadata completed for {}", source_name)
    logger.info("Step 1 completed for {}", source_name)
    return records, metadata
