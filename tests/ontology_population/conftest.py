"""Shared ontology-population test builders."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    PopulationCondition,
    PopulationConsequence,
    PopulationEvidence,
)


@pytest.fixture
def legal_chunk() -> Chunk:
    """Return one complete legal source chunk."""
    text = (
        "Người điều khiển xe mô tô không gắn biển số bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng. "
        "Phương tiện thuộc diện phải gắn biển số và đang tham gia giao thông. "
        "Bị trừ 06 điểm giấy phép lái xe. Bị tịch thu biển số. "
        "Buộc người điều khiển thực hiện đúng quy định về biển số."
    )
    return Chunk(
        chunk_id="chunk-1",
        content_sha256="content-sha",
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
        text=text,
        context_text=text,
        embedding_text=text,
        text_token_count=80,
        embedding_token_count=80,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


@pytest.fixture
def document_metadata() -> DocumentMetadataRecord:
    """Return reviewed document-level validity metadata."""
    return DocumentMetadataRecord(document_number="168/2024/NĐ-CP", valid_from=date(2025, 1, 1), valid_to=None)


@pytest.fixture
def population_record(legal_chunk: Chunk) -> OntologyPopulationRecord:
    """Return one complete record covering all populated module types."""
    rule_text = "Người điều khiển xe mô tô không gắn biển số bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."
    condition_text = "Phương tiện thuộc diện phải gắn biển số và đang tham gia giao thông."
    points_text = "Bị trừ 06 điểm giấy phép lái xe."
    sanction_text = "Bị tịch thu biển số."
    remedy_text = "Buộc người điều khiển thực hiện đúng quy định về biển số."
    return OntologyPopulationRecord(
        population_id="population-1",
        source_chunk_id=legal_chunk.chunk_id,
        conduct_label_vi="Điều khiển xe không gắn biển số",
        subject_kind="NaturalPerson",
        role_labels_vi=("Người điều khiển",),
        vehicle_category_labels_vi=("Xe mô tô",),
        condition_root_id="all",
        conditions=(
            PopulationCondition(
                condition_id="all",
                kind="AllOf",
                text_vi="Thuộc diện gắn biển số và đang tham gia giao thông",
                operand_ids=("required", "traffic"),
                evidence=(PopulationEvidence(legal_chunk.chunk_id, condition_text),),
            ),
            PopulationCondition(
                condition_id="required",
                kind="Atomic",
                text_vi="Thuộc diện phải gắn biển số",
                operand_ids=(),
                evidence=(PopulationEvidence(legal_chunk.chunk_id, condition_text),),
            ),
            PopulationCondition(
                condition_id="traffic",
                kind="Atomic",
                text_vi="Đang tham gia giao thông",
                operand_ids=(),
                evidence=(PopulationEvidence(legal_chunk.chunk_id, condition_text),),
            ),
        ),
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
                evidence=(PopulationEvidence(legal_chunk.chunk_id, rule_text),),
            ),
            PopulationConsequence(
                consequence_id="points",
                kind="PointDeduction",
                minimum_amount=None,
                maximum_amount=None,
                currency_code=None,
                point_value=6,
                target_label_vi="Giấy phép lái xe",
                responsible_role_label_vi=None,
                evidence=(PopulationEvidence(legal_chunk.chunk_id, points_text),),
            ),
            PopulationConsequence(
                consequence_id="sanction",
                kind="AdditionalSanction",
                minimum_amount=None,
                maximum_amount=None,
                currency_code=None,
                point_value=None,
                target_label_vi="Biển số",
                responsible_role_label_vi=None,
                evidence=(PopulationEvidence(legal_chunk.chunk_id, sanction_text),),
            ),
            PopulationConsequence(
                consequence_id="remedy",
                kind="RemedialMeasure",
                minimum_amount=None,
                maximum_amount=None,
                currency_code=None,
                point_value=None,
                target_label_vi="Biển số",
                responsible_role_label_vi="Người điều khiển",
                evidence=(PopulationEvidence(legal_chunk.chunk_id, remedy_text),),
            ),
        ),
        valid_from=date(2025, 1, 1),
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 13 khoản 9 điểm b",
        evidence=(PopulationEvidence(legal_chunk.chunk_id, rule_text),),
    )
