"""Typed intermediate records for ontology population."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from legal_rag.parser.hashing import sha256_json

ONTOLOGY_POPULATION_SCHEMA_VERSION = "0.2.0"
LEGACY_ONTOLOGY_POPULATION_SCHEMA_VERSION = "0.1.0"
ONTOLOGY_POPULATION_PROGRESS_SCHEMA_VERSION = "0.2.0"
LEGACY_ONTOLOGY_POPULATION_PROGRESS_SCHEMA_VERSION = "0.1.0"

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
    consequence_label_vi: str | None = None
    minimum_duration_months: int | None = None
    maximum_duration_months: int | None = None

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
            "consequence_label_vi": self.consequence_label_vi,
            "minimum_duration_months": self.minimum_duration_months,
            "maximum_duration_months": self.maximum_duration_months,
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
    ontology_version: str | None = None
    ontology_semantic_sha256: str | None = None
    population_run_id: str | None = None

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
            "ontology_version": self.ontology_version,
            "ontology_semantic_sha256": self.ontology_semantic_sha256,
            "population_run_id": self.population_run_id,
        }


@dataclass(frozen=True, slots=True)
class PopulationProgressRecord:
    """One target chunk whose ontology extraction completed."""

    chunk_id: str
    run_id: str | None

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable progress record."""
        if self.run_id is None:
            msg = "Current ontology population progress requires a run ID"
            raise ValueError(msg)
        return {
            "schema_version": ONTOLOGY_POPULATION_PROGRESS_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "run_id": self.run_id,
        }


def population_identity(content: dict[str, object]) -> str:
    """Return a deterministic identity for reviewed population content."""
    return sha256_json(content)


def validate_population_condition_tree(
    condition_root_id: str | None,
    conditions: tuple[PopulationCondition, ...],
) -> None:
    """Reject condition records that do not form one fully evidenced expression tree."""
    if not conditions:
        if condition_root_id is not None:
            msg = "A condition root requires at least one condition node"
            raise ValueError(msg)
        return
    if condition_root_id is None:
        msg = "Condition nodes require exactly one root ID"
        raise ValueError(msg)

    conditions_by_id = _index_population_conditions(conditions)
    if condition_root_id not in conditions_by_id:
        msg = f"Condition root does not exist: {condition_root_id}"
        raise ValueError(msg)
    for condition in conditions:
        _validate_population_condition(condition, conditions_by_id)

    visited = _reachable_population_condition_ids(condition_root_id, conditions_by_id)
    unreachable = set(conditions_by_id) - visited
    if unreachable:
        msg = f"Condition nodes are unreachable from root: {', '.join(sorted(unreachable))}"
        raise ValueError(msg)


def _index_population_conditions(
    conditions: tuple[PopulationCondition, ...],
) -> dict[str, PopulationCondition]:
    """Index one rule's condition nodes after validating their local identities."""
    condition_ids = [condition.condition_id for condition in conditions]
    if any(not condition_id for condition_id in condition_ids):
        msg = "Condition IDs must be non-empty"
        raise ValueError(msg)
    if len(condition_ids) != len(set(condition_ids)):
        msg = "Condition IDs must be unique within one population rule"
        raise ValueError(msg)
    return {condition.condition_id: condition for condition in conditions}


def _reachable_population_condition_ids(
    condition_root_id: str,
    conditions_by_id: dict[str, PopulationCondition],
) -> set[str]:
    """Return the root closure while rejecting recursive condition references."""
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(condition_id: str) -> None:
        if condition_id in visiting:
            msg = f"Condition tree contains a cycle at {condition_id}"
            raise ValueError(msg)
        if condition_id in visited:
            return
        visiting.add(condition_id)
        for operand_id in conditions_by_id[condition_id].operand_ids:
            visit(operand_id)
        visiting.remove(condition_id)
        visited.add(condition_id)

    visit(condition_root_id)
    return visited


def _validate_population_condition(
    condition: PopulationCondition,
    conditions_by_id: dict[str, PopulationCondition],
) -> None:
    """Validate one node's direct evidence, arity, and local references."""
    if condition.kind not in {"Atomic", "AllOf", "AnyOf", "Not"}:
        msg = f"Unsupported condition kind: {condition.kind}"
        raise ValueError(msg)
    if not condition.text_vi.strip():
        msg = f"Condition text must be non-empty: {condition.condition_id}"
        raise ValueError(msg)
    if not condition.evidence:
        msg = f"Every condition expression requires direct evidence: {condition.condition_id}"
        raise ValueError(msg)
    if len(condition.operand_ids) != len(set(condition.operand_ids)):
        msg = f"Condition operands must be unique: {condition.condition_id}"
        raise ValueError(msg)
    unknown_operands = set(condition.operand_ids) - set(conditions_by_id)
    if unknown_operands:
        msg = f"Condition operand does not exist: {sorted(unknown_operands)[0]}"
        raise ValueError(msg)

    operand_count = len(condition.operand_ids)
    if condition.kind == "Atomic" and operand_count != 0:
        msg = f"Atomic condition must not have operands: {condition.condition_id}"
        raise ValueError(msg)
    if condition.kind == "Not" and operand_count != 1:
        msg = f"Not condition must have exactly one operand: {condition.condition_id}"
        raise ValueError(msg)
    if condition.kind in {"AllOf", "AnyOf"} and operand_count < 2:
        msg = f"{condition.kind} condition must have at least two operands: {condition.condition_id}"
        raise ValueError(msg)


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
