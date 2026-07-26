"""Tests for atomic fact-ledger pipeline persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from legal_rag.evaluation.fact_ledger import FactLedgerEntry
from legal_rag.evaluation.models import EvidenceReference, LegalFact
from legal_rag.evaluation.writer import (
    read_fact_ledger,
    read_legal_facts,
    write_fact_ledger,
    write_legal_facts,
)


def _fact() -> LegalFact:
    """Build one legal fact covering every nullable and tuple field."""
    return LegalFact(
        fact_id="fact-1",
        fact_content_sha256="content-1",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=("Ban đêm",),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(
            EvidenceReference(
                chunk_id="chunk-1",
                quoted_span="Phạt tiền từ 4.000.000 đồng.",
                unit_ids=("unit-1",),
                source_refs=("word/document.xml#/body/1",),
            ),
        ),
    )


def test_legal_facts_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written legal fact reads back with identical field values."""
    output_path = tmp_path / "legal_facts.jsonl"
    fact = _fact()

    write_legal_facts(output_path, [fact])
    facts = read_legal_facts(output_path)

    assert facts == (fact,)


def test_fact_ledger_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written ledger entry reads back with identical field values, including timestamps."""
    output_path = tmp_path / "fact_ledger.jsonl"
    entry = FactLedgerEntry(
        fact_id="fact-1",
        fact_content_sha256="content-1",
        status="approved",
        reviewer="an.nguyen",
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    write_fact_ledger(output_path, [entry])
    entries = read_fact_ledger(output_path)

    assert entries == (entry,)


def test_fact_ledger_round_trip_preserves_null_reviewer(tmp_path: Path) -> None:
    """An unreviewed entry's null reviewer and timestamp survive the round trip."""
    output_path = tmp_path / "fact_ledger.jsonl"
    entry = FactLedgerEntry(
        fact_id="fact-1", fact_content_sha256="content-1", status="extracted", reviewer=None, reviewed_at=None
    )

    write_fact_ledger(output_path, [entry])
    entries = read_fact_ledger(output_path)

    assert entries == (entry,)


def test_read_fact_ledger_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """A ledger that has never been written reads back as no entries, not an error."""
    entries = read_fact_ledger(tmp_path / "does_not_exist.jsonl")

    assert entries == ()
