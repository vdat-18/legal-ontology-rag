"""Tests for the legal-fact schema migration command."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from legal_rag import legal_fact_migration_cli
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, build_legal_fact
from legal_rag.benchmark_synthesis.writer import read_legal_facts, write_document_metadata


def _stored_fact(**overrides: Any) -> dict[str, Any]:
    """Build one legal-fact payload as persisted under schema 1.1.0."""
    payload: dict[str, Any] = {
        "schema_version": "1.1.0",
        "fact_id": "stale-identity",
        "fact_content_sha256": "stale-content",
        "source_chunk_id": "chunk-1",
        "conduct": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "subject": "Người điều khiển ô tô",
        "conditions": [],
        "monetary_penalty": "4.000.000 đồng đến 6.000.000 đồng",
        "license_points_deducted": "4 điểm",
        "additional_penalties": [],
        "remedial_measures": [],
        "valid_from": None,
        "valid_to": None,
        "document_number": "",
        "provision_path": "Điều 6 khoản 9 điểm b",
        "evidence": [{"chunk_id": "chunk-1", "quoted_span": "Phạt tiền từ 4.000.000 đồng"}],
    }
    payload.update(overrides)
    return payload


def _write_inputs(
    tmp_path: Path,
    payloads: list[dict[str, Any]],
    *,
    metadata: DocumentMetadataRecord | None = None,
) -> tuple[Path, Path]:
    """Write one 1.1.0 facts file and its document-metadata file, returning both paths."""
    facts_path = tmp_path / "legal_facts.jsonl"
    metadata_path = tmp_path / "document_metadata.jsonl"
    facts_path.write_text(
        "".join(f"{json.dumps(payload, ensure_ascii=False)}\n" for payload in payloads), encoding="utf-8"
    )
    if metadata is None:
        metadata = DocumentMetadataRecord(document_number="168/2024/NĐ-CP", valid_from=date(2025, 1, 1), valid_to=None)
    write_document_metadata(metadata_path, metadata)
    return facts_path, metadata_path


def _run(monkeypatch: pytest.MonkeyPatch, facts_path: Path, metadata_path: Path) -> None:
    """Invoke the migration command against one document's artifacts."""
    monkeypatch.setattr(sys, "argv", ["legal-migrate-legal-facts", str(facts_path), str(metadata_path)])
    legal_fact_migration_cli.main()


def test_migration_stamps_document_number_and_effective_dates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An empty document number and absent effective date are replaced by the resolved document metadata."""
    facts_path, metadata_path = _write_inputs(tmp_path, [_stored_fact()])

    _run(monkeypatch, facts_path, metadata_path)

    facts = read_legal_facts(facts_path)
    assert len(facts) == 1
    assert facts[0].document_number == "168/2024/NĐ-CP"
    assert facts[0].valid_from == date(2025, 1, 1)
    assert facts[0].source_chunk_id == "chunk-1"


def test_migration_recomputes_identity_to_match_a_fresh_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A migrated fact carries exactly the identity a run with resolved metadata would have produced."""
    facts_path, metadata_path = _write_inputs(tmp_path, [_stored_fact()])
    expected = build_legal_fact(
        conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng đến 6.000.000 đồng",
        license_points_deducted="4 điểm",
        additional_penalties=(),
        remedial_measures=(),
        valid_from=date(2025, 1, 1),
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6 khoản 9 điểm b",
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span="Phạt tiền từ 4.000.000 đồng"),),
        source_chunk_id="chunk-1",
    )

    _run(monkeypatch, facts_path, metadata_path)

    migrated = read_legal_facts(facts_path)[0]
    assert migrated.fact_id == expected.fact_id
    assert migrated.fact_content_sha256 == expected.fact_content_sha256
    assert migrated.fact_id != "stale-identity"


def test_migration_keeps_a_provision_specific_effective_date(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A fact that already recorded its own effective date is not overwritten by the document's."""
    facts_path, metadata_path = _write_inputs(tmp_path, [_stored_fact(valid_from="2026-01-01")])

    _run(monkeypatch, facts_path, metadata_path)

    assert read_legal_facts(facts_path)[0].valid_from == date(2026, 1, 1)


def test_migration_merges_facts_that_share_a_recomputed_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two facts differing only by a stale document number collapse into the single fact they always were."""
    facts_path, metadata_path = _write_inputs(
        tmp_path,
        [_stored_fact(), _stored_fact(fact_id="other-identity", document_number="Nghị định này")],
    )

    _run(monkeypatch, facts_path, metadata_path)

    assert len(read_legal_facts(facts_path)) == 1


def test_migration_rejects_facts_written_under_another_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A facts file that is not schema 1.1.0 raises instead of being migrated twice."""
    facts_path, metadata_path = _write_inputs(tmp_path, [_stored_fact(schema_version="1.2.0")])

    with pytest.raises(ValueError, match=r"Expected legal-fact schema 1\.1\.0"):
        _run(monkeypatch, facts_path, metadata_path)
