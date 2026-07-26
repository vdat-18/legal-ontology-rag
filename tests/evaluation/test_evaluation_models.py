"""Tests for ground-truth synthesis data models."""

from __future__ import annotations

from datetime import date

from legal_rag.evaluation.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact


def _fact(**overrides: object) -> LegalFact:
    """Build one legal fact for serialization tests."""
    defaults: dict[str, object] = {
        "fact_id": "fact-id",
        "fact_content_sha256": "content-hash",
        "conduct": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "subject": "Người điều khiển ô tô",
        "conditions": (),
        "monetary_penalty": "4.000.000 đồng đến 6.000.000 đồng",
        "license_points_deducted": "4 điểm",
        "additional_penalties": (),
        "remedial_measures": (),
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
        "document_number": "168/2024/NĐ-CP",
        "provision_path": "Điều 6 khoản 9 điểm b",
        "evidence": (
            EvidenceReference(
                chunk_id="chunk-1",
                quoted_span="phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng",
                unit_ids=("unit-1",),
                source_refs=("word/document.xml#/body/1",),
            ),
        ),
    }
    defaults.update(overrides)
    return LegalFact(**defaults)


def test_legal_fact_to_dict_serializes_stable_output_schema() -> None:
    """Legal-fact tuples and dates serialize to JSON-compatible values."""
    record = _fact().to_dict()

    assert record["schema_version"] == LEGAL_FACT_SCHEMA_VERSION
    assert record["valid_from"] == "2025-01-01"
    assert record["valid_to"] is None
    assert record["conditions"] == []
    assert record["evidence"] == [
        {
            "chunk_id": "chunk-1",
            "quoted_span": "phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng",
            "unit_ids": ["unit-1"],
            "source_refs": ["word/document.xml#/body/1"],
        }
    ]


def test_legal_fact_to_dict_serializes_missing_penalty_fields_as_null() -> None:
    """Optional monetary and points fields serialize as null, not omitted."""
    record = _fact(monetary_penalty=None, license_points_deducted=None).to_dict()

    assert record["monetary_penalty"] is None
    assert record["license_points_deducted"] is None
