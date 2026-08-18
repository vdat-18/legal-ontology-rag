"""Load an explicitly ordered multi-document ontology corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import read_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import read_chunks
from legal_rag.ontology_population.models import OntologyPopulationRecord
from legal_rag.ontology_population.writer import read_population_records

CHANGE_CORPUS_MANIFEST_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ChangeCorpusDocument:
    """One ordered document and its canonical ontology-population artifacts."""

    document_id: str
    order_index: int
    chunks: tuple[Chunk, ...]
    metadata: DocumentMetadataRecord
    records: tuple[OntologyPopulationRecord, ...]


@dataclass(frozen=True, slots=True)
class ChangeCorpus:
    """One explicitly ordered corpus used for cross-document resolution."""

    documents: tuple[ChangeCorpusDocument, ...]


def read_change_corpus_manifest(path: Path) -> ChangeCorpus:
    """Load corpus artifacts from paths relative to a versioned JSON manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != CHANGE_CORPUS_MANIFEST_SCHEMA_VERSION:
        msg = f"Unsupported change corpus manifest: {payload['schema_version']}"
        raise ValueError(msg)
    documents = tuple(
        _load_document(path.parent, item)
        for item in sorted(payload["documents"], key=lambda item: (item["order_index"], item["document_id"]))
    )
    if len({document.document_id for document in documents}) != len(documents):
        msg = "Change corpus document identifiers must be unique"
        raise ValueError(msg)
    if len({document.order_index for document in documents}) != len(documents):
        msg = "Change corpus order indexes must be unique"
        raise ValueError(msg)
    if len({document.metadata.document_number for document in documents}) != len(documents):
        msg = "Change corpus document numbers must be unique"
        raise ValueError(msg)
    return ChangeCorpus(documents=documents)


def _load_document(manifest_directory: Path, payload: dict[str, object]) -> ChangeCorpusDocument:
    """Load one manifest document and enforce its document-number boundary."""
    chunks = read_chunks(manifest_directory / str(payload["chunks_path"]))
    metadata = read_document_metadata(manifest_directory / str(payload["metadata_path"]))
    records = read_population_records(manifest_directory / str(payload["canonical_records_path"]))
    if any(record.document_number != metadata.document_number for record in records):
        msg = "Canonical population records must match their manifest document metadata"
        raise ValueError(msg)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    if any(record.source_chunk_id not in chunk_ids for record in records):
        msg = "Canonical population records must reference chunks from their manifest document"
        raise ValueError(msg)
    return ChangeCorpusDocument(
        document_id=str(payload["document_id"]),
        order_index=int(payload["order_index"]),
        chunks=chunks,
        metadata=metadata,
        records=records,
    )
