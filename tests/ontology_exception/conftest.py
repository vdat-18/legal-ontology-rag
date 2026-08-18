"""Shared fixtures for legal-exception resolution tests."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import write_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    PopulationConsequence,
    PopulationEvidence,
)
from legal_rag.ontology_population.writer import write_population_records


@pytest.fixture
def exception_text() -> str:
    """Return one complete exact exception statement."""
    return "Trừ trường hợp quy định tại Điều 3."


@pytest.fixture
def exception_corpus(exception_text: str) -> ChangeCorpus:
    """Return one document containing a general rule and one redirected target rule."""
    general_text = f"Không chấp hành biển báo bị phạt tiền từ 400.000 đồng đến 600.000 đồng. {exception_text}"
    target_text = "Hành vi không chấp hành đèn tín hiệu bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."
    general_chunk = _chunk("general-chunk", 0, general_text)
    target_chunk = _chunk("target-chunk", 1, target_text)
    metadata = DocumentMetadataRecord(
        document_number="168/2024/NĐ-CP",
        valid_from=date(2025, 1, 1),
        valid_to=None,
    )
    general = _record(
        "general-population",
        general_chunk,
        metadata.document_number,
        "Không chấp hành biển báo",
        "Điều 6 khoản 1 điểm a",
        Decimal("400000"),
        Decimal("600000"),
    )
    target = _record(
        "target-population",
        target_chunk,
        metadata.document_number,
        "Không chấp hành đèn tín hiệu",
        "Điều 3",
        Decimal("4000000"),
        Decimal("6000000"),
    )
    return ChangeCorpus(
        documents=(
            ChangeCorpusDocument(
                document_id="nd-168",
                order_index=0,
                chunks=(general_chunk, target_chunk),
                metadata=metadata,
                records=(general, target),
            ),
        )
    )


def write_exception_manifest(path: Path, corpus: ChangeCorpus) -> Path:
    """Persist the fixture corpus manifest and return its flat canonical-record path."""
    document = corpus.documents[0]
    directory = path.parent / document.document_id
    chunks_path = directory / "chunks.jsonl"
    metadata_path = directory / "metadata.jsonl"
    manifest_records_path = directory / "manifest-records.jsonl"
    canonical_path = path.parent / "canonical.jsonl"
    write_chunks(chunks_path, document.chunks)
    write_document_metadata(metadata_path, document.metadata)
    write_population_records(manifest_records_path, document.records)
    write_population_records(canonical_path, document.records)
    payload = {
        "schema_version": "0.1.0",
        "documents": [
            {
                "document_id": document.document_id,
                "order_index": document.order_index,
                "chunks_path": str(chunks_path.relative_to(path.parent)),
                "metadata_path": str(metadata_path.relative_to(path.parent)),
                "canonical_records_path": str(manifest_records_path.relative_to(path.parent)),
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return canonical_path


def _chunk(chunk_id: str, index: int, text: str) -> Chunk:
    """Build one compact legal source chunk."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256=f"content-{chunk_id}",
        chunker_config_sha256="config-sha",
        scope_id="scope-168",
        chunk_index=index,
        kind="paragraph_group",
        source_name="168.docx",
        source_sha256="source-168",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=(f"unit-{chunk_id}",),
        source_refs=(f"document.xml#/{chunk_id}",),
        sequence_start=index,
        sequence_end=index,
        text=text,
        context_text=text,
        embedding_text=text,
        text_token_count=30,
        embedding_token_count=30,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def _record(
    population_id: str,
    chunk: Chunk,
    document_number: str,
    conduct: str,
    provision_path: str,
    minimum: Decimal,
    maximum: Decimal,
) -> OntologyPopulationRecord:
    """Build one complete monetary penalty rule."""
    evidence = (PopulationEvidence(chunk.chunk_id, chunk.text),)
    return OntologyPopulationRecord(
        population_id=population_id,
        source_chunk_id=chunk.chunk_id,
        conduct_label_vi=conduct,
        subject_kind="NaturalPerson",
        role_labels_vi=("Người điều khiển",),
        vehicle_category_labels_vi=("Xe mô tô",),
        condition_root_id=None,
        conditions=(),
        consequences=(
            PopulationConsequence(
                consequence_id="money",
                kind="MonetaryPenalty",
                minimum_amount=minimum,
                maximum_amount=maximum,
                currency_code="VND",
                point_value=None,
                target_label_vi=None,
                responsible_role_label_vi=None,
                evidence=evidence,
            ),
        ),
        valid_from=date(2025, 1, 1),
        valid_to=None,
        document_number=document_number,
        provision_path=provision_path,
        evidence=evidence,
    )
