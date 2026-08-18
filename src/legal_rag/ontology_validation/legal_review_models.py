"""Typed readiness report for independent ontology legal review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from legal_rag.ontology_validation.legal_review_config import LEGAL_REVIEW_REPORT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class LegalReviewCheckResult:
    """One structural legal-review packet check."""

    check_id: str
    status: Literal["passed", "failed"]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable check result."""
        return {"check_id": self.check_id, "status": self.status, "issues": list(self.issues)}


@dataclass(frozen=True, slots=True)
class LegalReviewReadinessReport:
    """Structural packet readiness and independent human-review progress."""

    checked_at: datetime
    artifact_id: str
    artifact_version: str
    overall_status: Literal["passed", "failed"]
    review_status: Literal["pending", "approved", "rejected", "changes_requested"]
    accepted_count: int
    total_count: int
    checks: tuple[LegalReviewCheckResult, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable readiness report."""
        return {
            "schema_version": LEGAL_REVIEW_REPORT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "overall_status": self.overall_status,
            "review_status": self.review_status,
            "accepted_count": self.accepted_count,
            "total_count": self.total_count,
            "checks": [check.to_dict() for check in self.checks],
        }
