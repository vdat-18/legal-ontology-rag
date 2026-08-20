"""Tests for ground-truth synthesis data models."""

from __future__ import annotations

from datetime import date

from legal_rag.benchmark_synthesis.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact


def _fact(**overrides: object) -> LegalFact:
    """Build one legal fact for serialization tests."""
    defaults: dict[str, object] = {
        "fact_id": "fact-id",
        "statement": "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền và trừ điểm.",
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
        "document_number": "168/2024/NĐ-CP",
        "evidence": (EvidenceReference(chunk_id="chunk-1"),),
    }
    defaults.update(overrides)
    return LegalFact(**defaults)


def test_legal_fact_to_dict_serializes_stable_output_schema() -> None:
    """Legal-fact tuples and dates serialize to JSON-compatible values."""
    record = _fact().to_dict()

    assert record["schema_version"] == LEGAL_FACT_SCHEMA_VERSION
    assert record["valid_from"] == "2025-01-01"
    assert record["valid_to"] is None
    assert record["statement"] == "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền và trừ điểm."
    assert record["evidence"] == [{"chunk_id": "chunk-1"}]
