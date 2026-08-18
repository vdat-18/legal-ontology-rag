"""Typed live-verification records for ontology publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

PUBLICATION_VERIFICATION_REPORT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class EndpointVerificationResult:
    """One live HTML, Turtle, or redirect verification result."""

    resource_id: str
    endpoint_kind: str
    representation: Literal["html", "turtle"]
    requested_url: str
    actual_status: int
    actual_media_type: str | None
    actual_location: str | None
    content_sha256: str
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable endpoint result."""
        return {
            "resource_id": self.resource_id,
            "endpoint_kind": self.endpoint_kind,
            "representation": self.representation,
            "requested_url": self.requested_url,
            "actual_status": self.actual_status,
            "actual_media_type": self.actual_media_type,
            "actual_location": self.actual_location,
            "content_sha256": self.content_sha256,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class PublicationVerificationReport:
    """One deployment-target or W3ID live-verification report."""

    checked_at: datetime
    mode: Literal["deployment", "w3id"]
    overall_status: Literal["passed", "failed"]
    results: tuple[EndpointVerificationResult, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable live-verification report."""
        return {
            "schema_version": PUBLICATION_VERIFICATION_REPORT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "mode": self.mode,
            "overall_status": self.overall_status,
            "results": [item.to_dict() for item in self.results],
        }
