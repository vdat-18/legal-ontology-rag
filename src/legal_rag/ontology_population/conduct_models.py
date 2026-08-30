"""Typed checkpoint records for target-conduct extraction."""

from __future__ import annotations

from dataclasses import dataclass

TARGET_CONDUCT_INVENTORY_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class TargetConductEvidenceReference:
    """One model-selected reference into target-chunk evidence."""

    evidence_id: str
    start_char: int | None
    end_char: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evidence reference."""
        return {
            "evidence_id": self.evidence_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True, slots=True)
class TargetConduct:
    """One atomic conduct identified directly in a target chunk."""

    conduct_id: str
    conduct_label_vi: str
    provision_path: str
    evidence: tuple[TargetConductEvidenceReference, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable conduct."""
        return {
            "conduct_id": self.conduct_id,
            "conduct_label_vi": self.conduct_label_vi,
            "provision_path": self.provision_path,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class TargetConductInventoryRecord:
    """Completed conduct inventory for one target chunk."""

    target_chunk_id: str
    conducts: tuple[TargetConduct, ...]
    run_id: str

    def to_dict(self) -> dict[str, object]:
        """Return the versioned persisted inventory schema."""
        return {
            "schema_version": TARGET_CONDUCT_INVENTORY_SCHEMA_VERSION,
            "target_chunk_id": self.target_chunk_id,
            "conducts": [item.to_dict() for item in self.conducts],
            "run_id": self.run_id,
        }
