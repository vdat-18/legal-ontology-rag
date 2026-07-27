"""Typed records for the fact-ledger ground-truth synthesis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

LEGAL_FACT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """One quoted span tying a legal fact to its exact chunk provenance."""

    chunk_id: str
    quoted_span: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence record."""
        return {
            "chunk_id": self.chunk_id,
            "quoted_span": self.quoted_span,
        }


@dataclass(frozen=True, slots=True)
class LegalFact:
    """One atomic legal fact with identity separated from its reviewable content."""

    fact_id: str
    fact_content_sha256: str
    conduct: str
    subject: str
    conditions: tuple[str, ...]
    monetary_penalty: str | None
    license_points_deducted: str | None
    additional_penalties: tuple[str, ...]
    remedial_measures: tuple[str, ...]
    valid_from: date | None
    valid_to: date | None
    document_number: str
    provision_path: str
    evidence: tuple[EvidenceReference, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-serializable legal-fact schema."""
        return {
            "schema_version": LEGAL_FACT_SCHEMA_VERSION,
            "fact_id": self.fact_id,
            "fact_content_sha256": self.fact_content_sha256,
            "conduct": self.conduct,
            "subject": self.subject,
            "conditions": list(self.conditions),
            "monetary_penalty": self.monetary_penalty,
            "license_points_deducted": self.license_points_deducted,
            "additional_penalties": list(self.additional_penalties),
            "remedial_measures": list(self.remedial_measures),
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "document_number": self.document_number,
            "provision_path": self.provision_path,
            "evidence": [entry.to_dict() for entry in self.evidence],
        }
