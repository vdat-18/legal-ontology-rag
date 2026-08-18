"""Typed process and report records for ontology semantic QA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

SEMANTIC_QA_REPORT_SCHEMA_VERSION = "0.2.0"

CheckStatus = Literal["passed", "failed", "blocked"]


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """One bounded external process result."""

    arguments: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class SemanticCheckResult:
    """One named release-hardening check."""

    check_id: str
    status: CheckStatus
    summary: str
    artifact: str | None
    exit_code: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable check result."""
        return {
            "check_id": self.check_id,
            "status": self.status,
            "summary": self.summary,
            "artifact": self.artifact,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True, slots=True)
class SemanticQaReport:
    """One versioned ontology semantic-QA result."""

    checked_at: datetime
    ontology_path: str
    catalog_path: str
    input_sha256: str
    tool_image: str
    profile: str
    reasoner: str
    overall_status: Literal["passed", "failed"]
    checks: tuple[SemanticCheckResult, ...]
    unsatisfiable_class_iris: tuple[str, ...]
    unexpected_equivalences: tuple[tuple[str, str], ...]
    traceability_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable semantic-QA schema."""
        return {
            "schema_version": SEMANTIC_QA_REPORT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "ontology_path": self.ontology_path,
            "catalog_path": self.catalog_path,
            "input_sha256": self.input_sha256,
            "tool_image": self.tool_image,
            "profile": self.profile,
            "reasoner": self.reasoner,
            "overall_status": self.overall_status,
            "checks": [item.to_dict() for item in self.checks],
            "unsatisfiable_class_iris": list(self.unsatisfiable_class_iris),
            "unexpected_equivalences": [list(item) for item in self.unexpected_equivalences],
            "traceability_issues": list(self.traceability_issues),
        }
