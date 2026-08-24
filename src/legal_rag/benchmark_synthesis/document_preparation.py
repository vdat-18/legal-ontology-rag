"""Persist screening and metadata artifacts for one document."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
) -> tuple[ChunkScreeningRecord, ...]:
    """Resume chunk screening and persist each completed batch."""
    existing_records = read_chunk_screening(output_path) if output_path.exists() else ()
    records_by_chunk_id = {record.chunk_id: record for record in existing_records}
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
            batch_size=settings.chunk_screening_batch_size,
            max_concurrency=settings.chunk_screening_max_concurrency,
            max_retries=settings.llm_max_retries,
            retry_delay_seconds=settings.llm_retry_delay_seconds,
            on_batch_screened=_persist_batch,
        )

    records = tuple(records_by_chunk_id[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in records_by_chunk_id)
    write_chunk_screening(output_path, records)
    return records


def prepare_document_metadata(
    chunks: Sequence[Chunk],
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
        client_factory(),
        opening_chunk_count=settings.document_metadata_opening_chunk_count,
        closing_chunk_count=settings.document_metadata_closing_chunk_count,
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
    )
    logger.info("Step 1A chunk screening completed for {}", source_name)
    logger.info("Step 1B document metadata started for {}", source_name)
    metadata = prepare_document_metadata(
        chunks,
        metadata_output_path,
        metadata_client_factory,
        settings,
    )
    logger.info("Step 1B document metadata completed for {}", source_name)
    logger.info("Step 1 completed for {}", source_name)
    return records, metadata
