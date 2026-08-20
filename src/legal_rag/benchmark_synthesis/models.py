"""Typed records for ground-truth synthesis legal facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from legal_rag.parser.hashing import sha256_json

LEGAL_FACT_SCHEMA_VERSION = "3.0.0"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """One source chunk supporting a legal fact."""

    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence record."""
        return {"chunk_id": self.chunk_id}


@dataclass(frozen=True, slots=True)
class LegalFact:
    """One self-contained legal statement with chunk-level provenance."""

    fact_id: str
    statement: str
    valid_from: date | None
    valid_to: date | None
    document_number: str
    evidence: tuple[EvidenceReference, ...]
    source_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-serializable legal-fact schema."""
        return {
            "schema_version": LEGAL_FACT_SCHEMA_VERSION,
            "fact_id": self.fact_id,
            "source_chunk_id": self.source_chunk_id,
            "statement": self.statement,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "document_number": self.document_number,
            "evidence": [entry.to_dict() for entry in self.evidence],
        }


def build_legal_fact(
    *,
    statement: str,
    valid_from: date | None,
    valid_to: date | None,
    document_number: str,
    evidence: tuple[EvidenceReference, ...],
    source_chunk_id: str | None,
) -> LegalFact:
    """Assign one content-derived identity to a legal statement and its provenance."""
    identity = {
        "statement": statement,
        "document_number": document_number,
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_to": valid_to.isoformat() if valid_to else None,
        "evidence": [entry.to_dict() for entry in evidence],
    }
    return LegalFact(
        fact_id=sha256_json(identity),
        statement=statement,
        valid_from=valid_from,
        valid_to=valid_to,
        document_number=document_number,
        evidence=evidence,
        source_chunk_id=source_chunk_id,
    )
