"""Tests for the fact-ledger approval state machine."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.evaluation.fact_ledger import (
    FACT_LEDGER_SCHEMA_VERSION,
    FactLedgerEntry,
    FactLedgerError,
    approve,
    queue_for_expert_review,
    reconcile_ledger,
    record_extracted,
    reject,
    verify_spans,
)
from legal_rag.evaluation.models import EvidenceReference, LegalFact


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one minimal chunk for span-verification tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="source.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def _fact(fact_id: str = "fact-1", fact_content_sha256: str = "content-1", quoted_span: str = "") -> LegalFact:
    """Build one minimal legal fact for ledger tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=fact_content_sha256,
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span=quoted_span, unit_ids=(), source_refs=()),),
    )


def test_record_extracted_creates_unreviewed_entry() -> None:
    """A freshly extracted fact starts unreviewed."""
    entry = record_extracted(_fact())

    assert entry.status == "extracted"
    assert entry.reviewer is None
    assert entry.reviewed_at is None


def test_verify_spans_accepts_matching_quote() -> None:
    """An exact quoted span moves the entry to span_verified."""
    chunk = _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _fact(quoted_span="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    entry = record_extracted(fact)

    verified = verify_spans(entry, fact, {"chunk-1": chunk}, threshold=0.85)

    assert verified.status == "span_verified"


def test_verify_spans_rejects_fabricated_quote() -> None:
    """A quote absent from the chunk text is rejected, not silently accepted."""
    chunk = _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _fact(quoted_span="Hoàn toàn không liên quan tới đoạn văn bản nguồn.")
    entry = record_extracted(fact)

    verified = verify_spans(entry, fact, {"chunk-1": chunk}, threshold=0.85)

    assert verified.status == "rejected"


def test_verify_spans_requires_extracted_status() -> None:
    """Span verification cannot run twice or out of order."""
    chunk = _chunk("chunk-1", "text")
    fact = _fact(quoted_span="text")
    entry = FactLedgerEntry(
        fact_id="fact-1", fact_content_sha256="content-1", status="approved", reviewer=None, reviewed_at=None
    )

    with pytest.raises(FactLedgerError):
        verify_spans(entry, fact, {"chunk-1": chunk}, threshold=0.85)


def test_queue_for_expert_review_requires_span_verified_status() -> None:
    """Queuing for review only follows a successful span check."""
    entry = record_extracted(_fact())

    with pytest.raises(FactLedgerError):
        queue_for_expert_review(entry)


def test_queue_for_expert_review_transitions_from_span_verified() -> None:
    """A span-verified entry moves into the expert review queue."""
    entry = replace_status(record_extracted(_fact()), "span_verified")

    queued = queue_for_expert_review(entry)

    assert queued.status == "needs_expert_review"


def test_approve_records_reviewer_and_timestamp() -> None:
    """Approval stamps the entry with who approved it and when."""
    entry = replace_status(record_extracted(_fact()), "needs_expert_review")
    reviewed_at = datetime(2026, 1, 1, tzinfo=UTC)

    approved = approve(entry, reviewer="an.nguyen", reviewed_at=reviewed_at)

    assert approved.status == "approved"
    assert approved.reviewer == "an.nguyen"
    assert approved.reviewed_at == reviewed_at


def test_reject_records_reviewer_and_timestamp() -> None:
    """Rejection stamps the entry with who rejected it and when."""
    entry = replace_status(record_extracted(_fact()), "needs_expert_review")
    reviewed_at = datetime(2026, 1, 1, tzinfo=UTC)

    rejected = reject(entry, reviewer="an.nguyen", reviewed_at=reviewed_at)

    assert rejected.status == "rejected"
    assert rejected.reviewer == "an.nguyen"


def test_approve_requires_needs_expert_review_status() -> None:
    """A fact cannot be approved before it reaches the review queue."""
    entry = record_extracted(_fact())

    with pytest.raises(FactLedgerError):
        approve(entry, reviewer="an.nguyen", reviewed_at=datetime(2026, 1, 1, tzinfo=UTC))


def test_reconcile_ledger_creates_entry_for_new_fact() -> None:
    """A fact with no existing ledger entry starts as extracted."""
    fact = _fact()

    reconciled = reconcile_ledger([fact], {})

    assert reconciled == (record_extracted(fact),)


def test_reconcile_ledger_preserves_unchanged_entry() -> None:
    """A fact whose content hash is unchanged keeps its existing entry untouched."""
    fact = _fact(fact_content_sha256="content-1")
    approved_entry = FactLedgerEntry(
        fact_id="fact-1",
        fact_content_sha256="content-1",
        status="approved",
        reviewer="an.nguyen",
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    reconciled = reconcile_ledger([fact], {"fact-1": approved_entry})

    assert reconciled == (approved_entry,)


def test_reconcile_ledger_marks_changed_approved_fact_as_stale() -> None:
    """A changed approved fact becomes stale, with its reviewer cleared."""
    fact = _fact(fact_content_sha256="content-2")
    approved_entry = FactLedgerEntry(
        fact_id="fact-1",
        fact_content_sha256="content-1",
        status="approved",
        reviewer="an.nguyen",
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    reconciled = reconcile_ledger([fact], {"fact-1": approved_entry})

    assert reconciled[0].status == "stale"
    assert reconciled[0].fact_content_sha256 == "content-2"
    assert reconciled[0].reviewer is None
    assert reconciled[0].reviewed_at is None


def test_reconcile_ledger_restarts_changed_unapproved_fact() -> None:
    """A changed fact that was never approved simply restarts at extracted."""
    fact = _fact(fact_content_sha256="content-2")
    pending_entry = FactLedgerEntry(
        fact_id="fact-1",
        fact_content_sha256="content-1",
        status="needs_expert_review",
        reviewer=None,
        reviewed_at=None,
    )

    reconciled = reconcile_ledger([fact], {"fact-1": pending_entry})

    assert reconciled == (record_extracted(fact),)


def test_fact_ledger_entry_to_dict_serializes_stable_output_schema() -> None:
    """Ledger entries serialize timestamps and nullable fields to JSON-compatible values."""
    entry = FactLedgerEntry(
        fact_id="fact-1",
        fact_content_sha256="content-1",
        status="approved",
        reviewer="an.nguyen",
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    record = entry.to_dict()

    assert record["schema_version"] == FACT_LEDGER_SCHEMA_VERSION
    assert record["reviewed_at"] == "2026-01-01T00:00:00+00:00"


def replace_status(entry: FactLedgerEntry, status: str) -> FactLedgerEntry:
    """Force an entry into an arbitrary status for setting up transition tests."""
    return replace(entry, status=status)
