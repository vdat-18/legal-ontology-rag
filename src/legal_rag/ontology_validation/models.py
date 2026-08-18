"""Typed records used by ontology validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModulePolicy:
    """One ontology module declared by the formalization policy."""

    module_id: str
    path: Path
    ontology_iri: str
    version_iri: str
    namespace: str
    imports: tuple[str, ...]


@dataclass(frozen=True)
class QueryCase:
    """One executable competency-question projection."""

    cq_id: str
    query_path: Path
    fixture_path: Path
    expected_columns: tuple[str, ...]
    expected_rows: int | None
    expected_minimum_rows: int | None
    deferred_output_fields: tuple[str, ...]


@dataclass(frozen=True)
class ValidationCase:
    """One SHACL fixture and its expected conformance result."""

    fixture_path: Path
    expected_conforms: bool


@dataclass(frozen=True)
class QueryResult:
    """Column names and materialized rows returned by a CQ query."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class ShaclResult:
    """Conformance status and report text returned by SHACL validation."""

    conforms: bool
    report_text: str
