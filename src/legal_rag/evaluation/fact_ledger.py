"""Approval state machine for legal facts, kept separate from their content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Literal

from rapidfuzz import fuzz

from legal_rag.chunker.models import Chunk
from legal_rag.evaluation.models import LegalFact

FACT_LEDGER_SCHEMA_VERSION = "1.0.0"

FactLedgerStatus = Literal[
    "extracted",
    "span_verified",
    "needs_expert_review",
    "approved",
    "rejected",
    "stale",
]

_TRANSITIONS: dict[str, FactLedgerStatus] = {
    "span_verified": "extracted",
    "needs_expert_review": "span_verified",
}


class FactLedgerError(ValueError):
    """Raised when a fact-ledger transition is attempted from an invalid status."""

    @classmethod
    def invalid_transition(cls, current_status: str, target_status: str) -> FactLedgerError:
        """Build an error for a transition that skips or repeats a required status."""
        return cls(f"Cannot move a fact from status {current_status!r} to {target_status!r}")


@dataclass(frozen=True, slots=True)
class FactLedgerEntry:
    """Approval state for one fact, tracked independently of its extracted content."""

    fact_id: str
    fact_content_sha256: str
    status: FactLedgerStatus
    reviewer: str | None
    reviewed_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable ledger record."""
        return {
            "schema_version": FACT_LEDGER_SCHEMA_VERSION,
            "fact_id": self.fact_id,
            "fact_content_sha256": self.fact_content_sha256,
            "status": self.status,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


def record_extracted(fact: LegalFact) -> FactLedgerEntry:
    """Create a new ledger entry for a freshly extracted fact."""
    return FactLedgerEntry(
        fact_id=fact.fact_id,
        fact_content_sha256=fact.fact_content_sha256,
        status="extracted",
        reviewer=None,
        reviewed_at=None,
    )


def _require_status(entry: FactLedgerEntry, target_status: FactLedgerStatus) -> None:
    """Raise unless the entry sits in the status required to reach target_status."""
    required_status = _TRANSITIONS[target_status]
    if entry.status != required_status:
        raise FactLedgerError.invalid_transition(entry.status, target_status)


def verify_spans(
    entry: FactLedgerEntry,
    fact: LegalFact,
    chunks_by_id: Mapping[str, Chunk],
    *,
    threshold: float,
) -> FactLedgerEntry:
    """Move an extracted entry to span_verified, or reject it on a failed quote match."""
    _require_status(entry, "span_verified")
    all_verified = all(
        fuzz.partial_ratio(evidence.quoted_span, chunks_by_id[evidence.chunk_id].text) >= threshold * 100
        for evidence in fact.evidence
    )
    return replace(entry, status="span_verified" if all_verified else "rejected")


def queue_for_expert_review(entry: FactLedgerEntry) -> FactLedgerEntry:
    """Move a span_verified entry into the expert review queue."""
    _require_status(entry, "needs_expert_review")
    return replace(entry, status="needs_expert_review")


def approve(entry: FactLedgerEntry, *, reviewer: str, reviewed_at: datetime) -> FactLedgerEntry:
    """Approve a fact awaiting expert review, recording who approved it and when."""
    if entry.status != "needs_expert_review":
        raise FactLedgerError.invalid_transition(entry.status, "approved")
    return replace(entry, status="approved", reviewer=reviewer, reviewed_at=reviewed_at)


def reject(entry: FactLedgerEntry, *, reviewer: str, reviewed_at: datetime) -> FactLedgerEntry:
    """Reject a fact awaiting expert review, recording who rejected it and when."""
    if entry.status != "needs_expert_review":
        raise FactLedgerError.invalid_transition(entry.status, "rejected")
    return replace(entry, status="rejected", reviewer=reviewer, reviewed_at=reviewed_at)


def reconcile_ledger(
    facts: Sequence[LegalFact],
    existing_entries: Mapping[str, FactLedgerEntry],
) -> tuple[FactLedgerEntry, ...]:
    """Merge freshly extracted facts into an existing ledger, preserving approval history.

    A fact whose content is unchanged keeps its existing entry untouched. A
    previously approved fact whose content changed becomes stale, with its
    reviewer cleared, since the approval no longer covers the new content. A
    fact that was never approved simply restarts at extracted; no approval
    decision exists yet to protect.
    """
    reconciled: list[FactLedgerEntry] = []
    for fact in facts:
        existing = existing_entries.get(fact.fact_id)
        if existing is None or existing.fact_content_sha256 != fact.fact_content_sha256:
            if existing is not None and existing.status == "approved":
                reconciled.append(
                    replace(
                        existing,
                        fact_content_sha256=fact.fact_content_sha256,
                        status="stale",
                        reviewer=None,
                        reviewed_at=None,
                    )
                )
            else:
                reconciled.append(record_extracted(fact))
        else:
            reconciled.append(existing)
    return tuple(reconciled)
