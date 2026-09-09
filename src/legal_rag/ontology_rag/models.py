"""Typed models for ontology-based RAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from rdflib import URIRef


@dataclass(frozen=True, slots=True)
class OntologyInstance:
    """One ontology instance linked to source chunks."""

    uri: URIRef
    type: str
    source_chunk_ids: tuple[str, ...]


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
    valid_from: date
    valid_to: date | None
    source_chunk_ids: tuple[str, ...]
    provision_path: str
    document_number: str
