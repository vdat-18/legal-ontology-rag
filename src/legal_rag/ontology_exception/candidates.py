"""Build exhaustive bounded windows for downstream exception confirmation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from legal_rag.chunker.models import Chunk
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.models import ExceptionCandidateSet, ExceptionCandidateWindow
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity

T = TypeVar("T")


def build_exception_candidates(
    corpus: ChangeCorpus,
    settings: ExceptionResolutionSettings,
) -> ExceptionCandidateSet:
    """Build exhaustive structural windows without classifying exception semantics."""
    all_targets = tuple(
        record for document in corpus.documents for record in _ordered_records(document.records, document.chunks)
    )
    target_batches = _batches(all_targets, settings.max_target_rules_per_window)
    windows = []
    for document in corpus.documents:
        source_batches = _batches(
            tuple(sorted(document.chunks, key=lambda item: (item.scope_id, item.chunk_index, item.chunk_id))),
            settings.max_source_chunks_per_window,
        )
        general_batches = _batches(
            _ordered_records(document.records, document.chunks),
            settings.max_general_rules_per_window,
        )
        for source_batch in source_batches:
            for general_batch in general_batches:
                for target_batch in target_batches:
                    windows.append(
                        _candidate_window(
                            document.document_id,
                            general_batch,
                            target_batch,
                            tuple(chunk.chunk_id for chunk in source_batch),
                        )
                    )
                    if len(windows) > settings.max_total_windows:
                        msg = "Exception candidate count exceeds the configured bound"
                        raise ValueError(msg)

    ordered_windows = tuple(sorted(windows, key=lambda window: window.candidate_id))
    run_id = population_identity(
        {
            "bounds": {
                "source_chunks": settings.max_source_chunks_per_window,
                "general_rules": settings.max_general_rules_per_window,
                "target_rules": settings.max_target_rules_per_window,
            },
            "documents": [
                {
                    "document_id": document.document_id,
                    "document_number": document.metadata.document_number,
                    "chunks": sorted(chunk.chunk_id for chunk in document.chunks),
                    "records": sorted(record.population_id for record in document.records),
                }
                for document in corpus.documents
            ],
        }
    )
    return ExceptionCandidateSet(run_id=run_id, windows=ordered_windows)


def _candidate_window(
    source_document_id: str,
    general_records: Sequence[OntologyPopulationRecord],
    target_records: Sequence[OntologyPopulationRecord],
    source_chunk_ids: tuple[str, ...],
) -> ExceptionCandidateWindow:
    """Build one stable structural exception window."""
    identity = {
        "source_document_id": source_document_id,
        "general_population_ids": tuple(record.population_id for record in general_records),
        "target_population_ids": tuple(record.population_id for record in target_records),
        "source_chunk_ids": source_chunk_ids,
    }
    return ExceptionCandidateWindow(
        candidate_id=population_identity(identity),
        source_document_id=source_document_id,
        general_population_ids=identity["general_population_ids"],
        target_population_ids=identity["target_population_ids"],
        source_chunk_ids=source_chunk_ids,
    )


def _ordered_records(
    records: Sequence[OntologyPopulationRecord],
    chunks: Sequence[Chunk],
) -> tuple[OntologyPopulationRecord, ...]:
    """Order records by source position and stable identity."""
    positions = {chunk.chunk_id: (chunk.scope_id, chunk.chunk_index) for chunk in chunks}
    return tuple(sorted(records, key=lambda record: (*positions[record.source_chunk_id], record.population_id)))


def _batches(items: Sequence[T], size: int) -> tuple[tuple[T, ...], ...]:
    """Partition a sequence into deterministic bounded batches."""
    return tuple(tuple(items[start : start + size]) for start in range(0, len(items), size))
