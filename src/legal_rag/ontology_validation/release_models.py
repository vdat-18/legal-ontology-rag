"""Typed inventory and report records for ontology releases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from legal_rag.ontology_validation.release_config import RELEASE_REPORT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    """One ontology release-readiness gate."""

    gate_id: str
    status: Literal["passed", "failed"]
    summary: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable gate result."""
        return {"gate_id": self.gate_id, "status": self.status, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class ReleaseFileRecord:
    """One immutable file in the release archive."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable file record."""
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class OntologyReleaseReport:
    """One durable ontology release or blocked-readiness result."""

    checked_at: datetime
    artifact_id: str
    artifact_version: str
    overall_status: Literal["released", "blocked"]
    gates: tuple[ReleaseGateResult, ...]
    files: tuple[ReleaseFileRecord, ...]
    archive_path: str | None
    archive_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable release report schema."""
        return {
            "schema_version": RELEASE_REPORT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "overall_status": self.overall_status,
            "gates": [item.to_dict() for item in self.gates],
            "files": [item.to_dict() for item in self.files],
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
        }
