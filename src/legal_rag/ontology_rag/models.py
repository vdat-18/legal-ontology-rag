"""Typed models for ontology-based RAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from rdflib import URIRef

ConditionKind = Literal["Atomic", "AllOf", "AnyOf", "Not"]
ConsequenceKind = Literal[
    "MonetaryPenalty",
    "Warning",
    "PointDeduction",
    "AdditionalSanction",
    "RemedialMeasure",
    "ReferencedPenalty",
]


@dataclass(frozen=True, slots=True)
class OntologyInstance:
    """One ontology instance linked to source chunks."""

    uri: URIRef
    type: str
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComponentProvenance:
    """Source provisions and chunks supporting one rule component."""

    provision_uris: tuple[URIRef, ...] = ()
    provision_paths: tuple[str, ...] = ()
    source_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OntologyTerm:
    """One URI-valued ontology term and its preferred Vietnamese label."""

    uri: URIRef
    label_vi: str | None


@dataclass(frozen=True, slots=True)
class ConditionExpression:
    """One source-grounded node in an applicability condition tree."""

    uri: URIRef
    kind: ConditionKind
    label_vi: str | None
    operands: tuple[ConditionExpression, ...] = ()
    provenance: ComponentProvenance = ComponentProvenance()


@dataclass(frozen=True, slots=True)
class LegalConsequence:
    """One independently evidenced legal consequence."""

    uri: URIRef
    kind: ConsequenceKind
    label_vi: str | None = None
    minimum_amount: Decimal | None = None
    maximum_amount: Decimal | None = None
    currency: str | None = None
    point_value: int | None = None
    minimum_duration_months: int | None = None
    maximum_duration_months: int | None = None
    affected_targets: tuple[OntologyTerm, ...] = ()
    responsible_roles: tuple[OntologyTerm, ...] = ()
    provenance: ComponentProvenance = ComponentProvenance()


@dataclass(frozen=True, slots=True)
class ProvisionCitation:
    """One grouped provision citation preserved by a legal exception."""

    uri: URIRef
    citation_text_vi: str | None = None
    referenced_provision_uris: tuple[URIRef, ...] = ()
    referenced_provision_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LegalException:
    """One evidenced exception and its verified redirect targets."""

    uri: URIRef
    citations: tuple[ProvisionCitation, ...] = ()
    excluded_provision_uris: tuple[URIRef, ...] = ()
    excluded_provision_paths: tuple[str, ...] = ()
    redirected_rule_uris: tuple[URIRef, ...] = ()
    provenance: ComponentProvenance = ComponentProvenance()


@dataclass(frozen=True, slots=True)
class PenaltyRule:
    """One populated penalty rule with full properties."""

    uri: URIRef
    conduct_label_vi: str
    vehicle_categories: tuple[str, ...]
    subject_kind: str | None
    roles: tuple[str, ...]
    monetary_penalty_min: Decimal | None
    monetary_penalty_max: Decimal | None
    currency: str | None
    point_value: int | None
    valid_from: date | None
    valid_to: date | None
    source_chunk_ids: tuple[str, ...]
    provision_path: str
    document_number: str
    consequences: tuple[LegalConsequence, ...] = ()
    condition_expressions: tuple[ConditionExpression, ...] = ()
    exceptions: tuple[LegalException, ...] = ()
    rule_provenance: ComponentProvenance = ComponentProvenance()
    conduct_provenance: tuple[ComponentProvenance, ...] = ()
    applicability_provenance: tuple[ComponentProvenance, ...] = ()
