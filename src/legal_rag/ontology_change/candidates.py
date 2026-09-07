"""Build bounded provenance-aligned windows for legal-change confirmation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from legal_rag.chunker.models import Chunk
from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.models import ChangeCandidateSet, ChangeCandidateWindow
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity

T = TypeVar("T")


def build_change_candidates(corpus: ChangeCorpus, settings: ChangeResolutionSettings) -> ChangeCandidateSet:
    """Build ordered cross-document windows without making a legal-change decision."""
    windows = []
    for later_index, later_document in enumerate(corpus.documents):
        later_chunks = tuple(
            sorted(later_document.chunks, key=lambda item: (item.scope_id, item.chunk_index, item.chunk_id))
        )
        later_records = _ordered_records(later_document.records, later_document.chunks)
        source_batches = _batches(later_chunks, settings.max_source_chunks_per_window)
        earlier_documents = corpus.documents[:later_index]
        if later_document.comparison_document_ids is not None:
            comparison_ids = set(later_document.comparison_document_ids)
            earlier_documents = tuple(
                document for document in earlier_documents if document.document_id in comparison_ids
            )
        for earlier_document in earlier_documents:
            earlier_records = _ordered_records(earlier_document.records, earlier_document.chunks)
            for source_batch in source_batches:
                source_chunk_ids = tuple(chunk.chunk_id for chunk in source_batch)
                source_chunk_id_set = set(source_chunk_ids)
                source_records = tuple(
                    record for record in later_records if record.source_chunk_id in source_chunk_id_set
                )
                later_batches = _batches(source_records, settings.max_later_records_per_window) or ((),)
                for earlier_batch in _batches(earlier_records, settings.max_earlier_records_per_window):
                    for later_batch in later_batches:
                        windows.append(
                            _candidate_window(
                                earlier_document.document_id,
                                later_document.document_id,
                                earlier_batch,
                                later_batch,
                                source_chunk_ids,
                            )
                        )
                        if len(windows) > settings.max_total_windows:
                            msg = "Cross-document candidate count exceeds the configured bound"
                            raise ValueError(msg)

    ordered_windows = tuple(sorted(windows, key=lambda window: window.candidate_id))
    run_id = population_identity(
        {
            "bounds": {
                "source_chunks": settings.max_source_chunks_per_window,
                "earlier_records": settings.max_earlier_records_per_window,
                "later_records": settings.max_later_records_per_window,
            },
            "later_record_scope": "change-source-batch",
            "documents": [
                {
                    "document_id": document.document_id,
                    "order_index": document.order_index,
                    "document_number": document.metadata.document_number,
                    "valid_from": (document.metadata.valid_from.isoformat() if document.metadata.valid_from else None),
                    "valid_to": document.metadata.valid_to.isoformat() if document.metadata.valid_to else None,
                    "comparison_document_ids": document.comparison_document_ids,
                    "chunks": sorted(chunk.chunk_id for chunk in document.chunks),
                    "records": sorted(record.population_id for record in document.records),
                }
                for document in corpus.documents
            ],
        }
    )
    return ChangeCandidateSet(run_id=run_id, windows=ordered_windows)


def _candidate_window(
    earlier_document_id: str,
    later_document_id: str,
    earlier_records: Sequence[OntologyPopulationRecord],
    later_records: Sequence[OntologyPopulationRecord],
    source_chunk_ids: tuple[str, ...],
) -> ChangeCandidateWindow:
    """Build one stable structural window."""
    earlier_ids = tuple(record.population_id for record in earlier_records)
    later_ids = tuple(record.population_id for record in later_records)
    identity = {
        "earlier_document_id": earlier_document_id,
        "later_document_id": later_document_id,
        "earlier_population_ids": earlier_ids,
        "later_population_ids": later_ids,
        "change_source_chunk_ids": source_chunk_ids,
    }
    return ChangeCandidateWindow(
        candidate_id=population_identity(identity),
        earlier_document_id=earlier_document_id,
        later_document_id=later_document_id,
        earlier_population_ids=earlier_ids,
        later_population_ids=later_ids,
        change_source_chunk_ids=source_chunk_ids,
    )


def _ordered_records(
    records: Sequence[OntologyPopulationRecord], chunks: Sequence[Chunk]
) -> tuple[OntologyPopulationRecord, ...]:
    """Order records by their source chunk position and stable identity."""
    positions = {chunk.chunk_id: (chunk.scope_id, chunk.chunk_index) for chunk in chunks}
    return tuple(sorted(records, key=lambda record: (*positions[record.source_chunk_id], record.population_id)))


def _batches(items: Sequence[T], size: int) -> tuple[tuple[T, ...], ...]:
    """Partition a sequence into deterministic bounded batches."""
    return tuple(tuple(items[start : start + size]) for start in range(0, len(items), size))
