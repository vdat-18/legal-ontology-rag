"""Typed intermediate records for ontology population."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from legal_rag.parser.hashing import sha256_json

ONTOLOGY_POPULATION_SCHEMA_VERSION = "0.1.0"
ONTOLOGY_POPULATION_PROGRESS_SCHEMA_VERSION = "0.1.0"

SubjectKind = Literal["NaturalPerson", "Organization"]
ConditionKind = Literal["Atomic", "AllOf", "AnyOf", "Not"]
ConsequenceKind = Literal["MonetaryPenalty", "PointDeduction", "AdditionalSanction", "RemedialMeasure"]
CurrencyCode = Literal["VND"]


@dataclass(frozen=True, slots=True)
class PopulationEvidence:
    """One exact source span supporting a populated assertion."""

    chunk_id: str
    exact_text: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable evidence reference."""
        return {"chunk_id": self.chunk_id, "exact_text": self.exact_text}


@dataclass(frozen=True, slots=True)
class PopulationCondition:
    """One node in a source-evidenced condition-expression tree."""

    condition_id: str
    kind: ConditionKind
    text_vi: str
    operand_ids: tuple[str, ...]
    evidence: tuple[PopulationEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable condition node."""
        return {
            "condition_id": self.condition_id,
            "kind": self.kind,
            "text_vi": self.text_vi,
            "operand_ids": list(self.operand_ids),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class PopulationConsequence:
    """One source-classified legal consequence."""

    consequence_id: str
    kind: ConsequenceKind
    minimum_amount: Decimal | None
    maximum_amount: Decimal | None
    currency_code: CurrencyCode | None
    point_value: int | None
    target_label_vi: str | None
    responsible_role_label_vi: str | None
    evidence: tuple[PopulationEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable consequence record."""
        return {
            "consequence_id": self.consequence_id,
            "kind": self.kind,
            "minimum_amount": str(self.minimum_amount) if self.minimum_amount is not None else None,
            "maximum_amount": str(self.maximum_amount) if self.maximum_amount is not None else None,
            "currency_code": self.currency_code,
            "point_value": self.point_value,
            "target_label_vi": self.target_label_vi,
            "responsible_role_label_vi": self.responsible_role_label_vi,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class OntologyPopulationRecord:
    """One version-specific penalty rule ready for RDF materialization."""

    population_id: str
    source_chunk_id: str
    conduct_label_vi: str
    subject_kind: SubjectKind | None
    role_labels_vi: tuple[str, ...]
    vehicle_category_labels_vi: tuple[str, ...]
    condition_root_id: str | None
    conditions: tuple[PopulationCondition, ...]
    consequences: tuple[PopulationConsequence, ...]
    valid_from: date | None
    valid_to: date | None
    document_number: str
    provision_path: str
    evidence: tuple[PopulationEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-serializable population schema."""
        return {
            "schema_version": ONTOLOGY_POPULATION_SCHEMA_VERSION,
            "population_id": self.population_id,
            "source_chunk_id": self.source_chunk_id,
            "conduct_label_vi": self.conduct_label_vi,
            "subject_kind": self.subject_kind,
            "role_labels_vi": list(self.role_labels_vi),
            "vehicle_category_labels_vi": list(self.vehicle_category_labels_vi),
            "condition_root_id": self.condition_root_id,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "consequences": [consequence.to_dict() for consequence in self.consequences],
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "document_number": self.document_number,
            "provision_path": self.provision_path,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class PopulationProgressRecord:
    """One target chunk whose ontology extraction completed."""

    chunk_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable progress record."""
        return {
            "schema_version": ONTOLOGY_POPULATION_PROGRESS_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
        }


def population_identity(content: dict[str, object]) -> str:
    """Return a deterministic identity for reviewed population content."""
    return sha256_json(content)


def with_population_validity(
    record: OntologyPopulationRecord,
    *,
    valid_from: date | None,
    valid_to: date | None,
) -> OntologyPopulationRecord:
    """Return a reidentified population record with reviewed validity dates."""
    updated = replace(record, population_id="", valid_from=valid_from, valid_to=valid_to)
    content = updated.to_dict()
    del content["schema_version"]
    del content["population_id"]
    return replace(updated, population_id=population_identity(content))
