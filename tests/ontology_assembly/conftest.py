"""Shared cross-chunk assembly fixtures."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    PopulationConsequence,
    PopulationEvidence,
)


@pytest.fixture
def assembly_chunks() -> tuple[Chunk, Chunk]:
    """Return adjacent chunks containing two parts of one legal rule."""
    first_text = "Người điều khiển xe mô tô không gắn biển số bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."
    second_text = "Thực hiện hành vi này còn bị trừ 06 điểm giấy phép lái xe."
    base = Chunk(
        chunk_id="chunk-1",
        content_sha256="content-1",
        chunker_config_sha256="config-sha",
        scope_id="scope-1",
        chunk_index=0,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
        source_sha256="source-sha",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-1",),
        source_refs=("document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=first_text,
        context_text=first_text,
        embedding_text=first_text,
        text_token_count=30,
        embedding_token_count=30,
        previous_chunk_id=None,
        next_chunk_id="chunk-2",
    )
    second = replace(
        base,
        chunk_id="chunk-2",
        content_sha256="content-2",
        chunk_index=1,
        unit_ids=("unit-2",),
        source_refs=("document.xml#/body/1",),
        sequence_start=1,
        sequence_end=1,
        text=second_text,
        context_text=second_text,
        embedding_text=second_text,
        previous_chunk_id="chunk-1",
        next_chunk_id=None,
    )
    return base, second


@pytest.fixture
def assembly_metadata() -> DocumentMetadataRecord:
    """Return reviewed metadata for the assembly source document."""
    return DocumentMetadataRecord(document_number="168/2024/NĐ-CP", valid_from=date(2025, 1, 1), valid_to=None)


@pytest.fixture
def assembly_records(assembly_chunks: tuple[Chunk, Chunk]) -> tuple[OntologyPopulationRecord, ...]:
    """Return two source records that may be assembled into one rule."""
    first, second = assembly_chunks
    common = {
        "conduct_label_vi": "Điều khiển xe không gắn biển số",
        "subject_kind": "NaturalPerson",
        "role_labels_vi": ("Người điều khiển",),
        "vehicle_category_labels_vi": ("Xe mô tô",),
        "condition_root_id": None,
        "conditions": (),
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
        "document_number": "168/2024/NĐ-CP",
    }
    monetary = OntologyPopulationRecord(
        population_id="population-money",
        source_chunk_id=first.chunk_id,
        consequences=(
            PopulationConsequence(
                consequence_id="money",
                kind="MonetaryPenalty",
                minimum_amount=Decimal("4000000"),
                maximum_amount=Decimal("6000000"),
                currency_code="VND",
                point_value=None,
                target_label_vi=None,
                responsible_role_label_vi=None,
                evidence=(PopulationEvidence(first.chunk_id, first.text),),
            ),
        ),
        provision_path="Điều 13 khoản 9 điểm b",
        evidence=(PopulationEvidence(first.chunk_id, first.text),),
        **common,
    )
    points = OntologyPopulationRecord(
        population_id="population-points",
        source_chunk_id=second.chunk_id,
        consequences=(
            PopulationConsequence(
                consequence_id="points",
                kind="PointDeduction",
                minimum_amount=None,
                maximum_amount=None,
                currency_code=None,
                point_value=6,
                target_label_vi="Giấy phép lái xe",
                responsible_role_label_vi=None,
                evidence=(PopulationEvidence(second.chunk_id, second.text),),
            ),
        ),
        provision_path="Điều 13 khoản 9 điểm b",
        evidence=(PopulationEvidence(second.chunk_id, second.text),),
        **common,
    )
    return monetary, points
