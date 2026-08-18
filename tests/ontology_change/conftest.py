"""Shared fixtures for cross-document legal-change tests."""

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
def change_documents() -> tuple[ChangeCorpusDocument, ChangeCorpusDocument]:
    """Return an earlier rule and an explicitly amending later document."""
    earlier_text = "Người điều khiển xe mô tô không gắn biển số bị phạt tiền từ 3.000.000 đồng đến 4.000.000 đồng."
    change_text = "Sửa đổi điểm b khoản 8 Điều 13, có hiệu lực từ ngày 15 tháng 8 năm 2026."
    later_rule_text = "Người điều khiển xe mô tô không gắn biển số bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."
    earlier_chunk = _chunk("earlier-chunk", "earlier-source", "scope-earlier", 0, earlier_text, "168.docx")
    later_chunk = _chunk(
        "later-chunk",
        "later-source",
        "scope-later",
        0,
        f"{change_text} {later_rule_text}",
        "238.docx",
    )
    earlier_metadata = DocumentMetadataRecord(
        document_number="168/2024/NĐ-CP",
        valid_from=date(2025, 1, 1),
        valid_to=None,
    )
    later_metadata = DocumentMetadataRecord(
        document_number="238/2026/NĐ-CP",
        valid_from=date(2026, 8, 15),
        valid_to=None,
    )
    earlier_record = _record(
        "earlier-population",
        earlier_chunk,
        earlier_metadata.document_number,
        Decimal("3000000"),
        Decimal("4000000"),
        date(2025, 1, 1),
    )
    later_record = _record(
        "later-population",
        later_chunk,
        later_metadata.document_number,
        Decimal("4000000"),
        Decimal("6000000"),
        None,
        evidence_text=later_rule_text,
    )
    return (
        ChangeCorpusDocument(
            document_id="nd-168",
            order_index=0,
            chunks=(earlier_chunk,),
            metadata=earlier_metadata,
            records=(earlier_record,),
        ),
        ChangeCorpusDocument(
            document_id="nd-238",
            order_index=1,
            chunks=(later_chunk,),
            metadata=later_metadata,
            records=(later_record,),
        ),
    )


@pytest.fixture
def change_corpus(change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument]) -> ChangeCorpus:
    """Return one explicitly ordered two-document corpus."""
    return ChangeCorpus(documents=change_documents)


@pytest.fixture
def change_evidence_text() -> str:
    """Return the exact authoritative amendment statement."""
    return "Sửa đổi điểm b khoản 8 Điều 13, có hiệu lực từ ngày 15 tháng 8 năm 2026."


def write_manifest(path: Path, documents: tuple[ChangeCorpusDocument, ...]) -> None:
    """Persist a self-contained ordered corpus manifest and its artifacts."""
    entries = []
    for document in documents:
        directory = path.parent / document.document_id
        chunks_path = directory / "chunks.jsonl"
        metadata_path = directory / "metadata.jsonl"
        records_path = directory / "canonical.jsonl"
        write_chunks(chunks_path, document.chunks)
        write_document_metadata(metadata_path, document.metadata)
        write_population_records(records_path, document.records)
        entries.append(
            {
                "document_id": document.document_id,
                "order_index": document.order_index,
                "chunks_path": str(chunks_path.relative_to(path.parent)),
                "metadata_path": str(metadata_path.relative_to(path.parent)),
                "canonical_records_path": str(records_path.relative_to(path.parent)),
            }
        )
    path.write_text(
        json.dumps({"schema_version": "0.1.0", "documents": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _chunk(chunk_id: str, source_sha: str, scope_id: str, index: int, text: str, source_name: str) -> Chunk:
    """Build one compact legal source chunk."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256=f"content-{chunk_id}",
        chunker_config_sha256="config-sha",
        scope_id=scope_id,
        chunk_index=index,
        kind="paragraph_group",
        source_name=source_name,
        source_sha256=source_sha,
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
        text_token_count=40,
        embedding_token_count=40,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def _record(
    population_id: str,
    chunk: Chunk,
    document_number: str,
    minimum: Decimal,
    maximum: Decimal,
    valid_from: date | None,
    *,
    evidence_text: str | None = None,
) -> OntologyPopulationRecord:
    """Build one complete monetary penalty rule version."""
    exact_text = evidence_text or chunk.text
    evidence = (PopulationEvidence(chunk.chunk_id, exact_text),)
    return OntologyPopulationRecord(
        population_id=population_id,
        source_chunk_id=chunk.chunk_id,
        conduct_label_vi="Điều khiển xe không gắn biển số",
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
        valid_from=valid_from,
        valid_to=None,
        document_number=document_number,
        provision_path="Điều 13 khoản 8 điểm b",
        evidence=evidence,
    )
