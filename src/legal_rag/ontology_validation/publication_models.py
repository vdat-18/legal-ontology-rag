"""Typed reports for ontology namespace publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from legal_rag.ontology_validation.publication_config import PUBLICATION_REPORT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PublicationCheckResult:
    """One namespace-publication readiness check."""

    check_id: str
    status: Literal["passed", "failed"]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable check result."""
        return {"check_id": self.check_id, "status": self.status, "issues": list(self.issues)}


@dataclass(frozen=True, slots=True)
class PublishedFileRecord:
    """One generated namespace-publication file."""

    path: str
    media_type: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable publication file record."""
        return {
            "path": self.path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class NamespacePublicationReport:
    """One durable namespace-publication build result."""

    checked_at: datetime
    artifact_id: str
    artifact_version: str
    overall_status: Literal["built", "failed"]
    checks: tuple[PublicationCheckResult, ...]
    files: tuple[PublishedFileRecord, ...]
    manifest_path: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable publication report."""
        return {
            "schema_version": PUBLICATION_REPORT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "overall_status": self.overall_status,
            "checks": [item.to_dict() for item in self.checks],
            "files": [item.to_dict() for item in self.files],
            "manifest_path": self.manifest_path,
        }
