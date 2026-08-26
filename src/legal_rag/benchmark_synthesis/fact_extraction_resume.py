"""Validate and prepare resumable fact-extraction artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord
from legal_rag.benchmark_synthesis.fact_extraction_selection import validate_selective_extraction
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.stage_cache import (
    StageCacheFingerprint,
    initialize_stage_cache,
    read_stage_cache_fingerprint,
    require_fact_extraction_cache,
    write_stage_cache_fingerprint,
)
from legal_rag.benchmark_synthesis.writer import write_extraction_progress, write_legal_facts
from legal_rag.chunker.models import Chunk


def prepare_extraction_resume(
    *,
    fingerprint_path: Path,
    facts_path: Path,
    progress_path: Path,
    chunks: Sequence[Chunk],
    accepted_chunks: Sequence[Chunk],
    metadata: DocumentMetadataRecord,
    existing_facts: Sequence[LegalFact],
    existing_progress: Sequence[ExtractionProgressRecord],
    execution_fingerprint: StageCacheFingerprint,
    reextract_chunk_ids: frozenset[str],
) -> tuple[tuple[LegalFact, ...], tuple[ExtractionProgressRecord, ...]]:
    """Validate cache identity, migrate provenance, and invalidate selected targets."""
    accepted_chunk_id_order = tuple(chunk.chunk_id for chunk in accepted_chunks)
    existing_fingerprint = _validate_cache(
        fingerprint_path=fingerprint_path,
        artifact_paths=(facts_path, progress_path),
        chunks=chunks,
        accepted_chunk_id_order=accepted_chunk_id_order,
        metadata=metadata,
        existing_facts=existing_facts,
        existing_progress=existing_progress,
        execution_fingerprint=execution_fingerprint,
        reextract_chunk_ids=reextract_chunk_ids,
    )
    progress = tuple(
        ExtractionProgressRecord(
            chunk_id=record.chunk_id,
            prompt_version=record.prompt_version or (existing_fingerprint.prompt_version if existing_fingerprint else None),
            model=record.model or (existing_fingerprint.model if existing_fingerprint else None),
            extraction_run_id=record.extraction_run_id or (existing_fingerprint.run_id if existing_fingerprint else None),
        )
        for record in existing_progress
        if record.chunk_id not in reextract_chunk_ids
    )
    facts = tuple(fact for fact in existing_facts if fact.source_chunk_id not in reextract_chunk_ids)
    if reextract_chunk_ids:
        write_legal_facts(facts_path, facts)
        write_extraction_progress(progress_path, progress)
        write_stage_cache_fingerprint(fingerprint_path, execution_fingerprint)
    return facts, progress


def _validate_cache(
    *,
    fingerprint_path: Path,
    artifact_paths: tuple[Path, Path],
    chunks: Sequence[Chunk],
    accepted_chunk_id_order: tuple[str, ...],
    metadata: DocumentMetadataRecord,
    existing_facts: Sequence[LegalFact],
    existing_progress: Sequence[ExtractionProgressRecord],
    execution_fingerprint: StageCacheFingerprint,
    reextract_chunk_ids: frozenset[str],
) -> StageCacheFingerprint | None:
    """Return the prior cache identity after validating normal or selective resume."""
    if reextract_chunk_ids:
        if not fingerprint_path.exists() or not all(path.exists() for path in artifact_paths):
            msg = "Selective fact re-extraction requires existing facts, progress, and run fingerprint"
            raise ValueError(msg)
        validate_selective_extraction(
            frozenset(accepted_chunk_id_order),
            existing_progress,
            reextract_chunk_ids,
        )
        return read_stage_cache_fingerprint(fingerprint_path)
    if fingerprint_path.exists() and any(path.exists() for path in artifact_paths):
        if existing_progress and all(record.extraction_run_id for record in existing_progress):
            return require_fact_extraction_cache(
                fingerprint_path,
                chunks,
                accepted_chunk_id_order,
                metadata,
                existing_facts,
                existing_progress,
                execution_fingerprint,
            )
        initialize_stage_cache(fingerprint_path, artifact_paths, execution_fingerprint)
        return read_stage_cache_fingerprint(fingerprint_path)
    initialize_stage_cache(fingerprint_path, artifact_paths, execution_fingerprint)
    return None
