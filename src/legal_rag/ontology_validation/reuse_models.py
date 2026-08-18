"""Typed report records for external ontology reuse review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from legal_rag.ontology_validation.reuse_config import REUSE_REPORT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ReuseCheckResult:
    """One structural external-reuse review check."""

    check_id: str
    status: Literal["passed", "failed"]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable check result."""
        return {"check_id": self.check_id, "status": self.status, "issues": list(self.issues)}


@dataclass(frozen=True, slots=True)
class ExternalReuseReport:
    """One versioned technical external-reuse review result."""

    checked_at: datetime
    artifact_id: str
    artifact_version: str
    overall_status: Literal["passed", "failed"]
    project_owner_approval: str
    checks: tuple[ReuseCheckResult, ...]
    active_external_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable reuse-review report."""
        return {
            "schema_version": REUSE_REPORT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "overall_status": self.overall_status,
            "project_owner_approval": self.project_owner_approval,
            "checks": [item.to_dict() for item in self.checks],
            "active_external_terms": list(self.active_external_terms),
        }
