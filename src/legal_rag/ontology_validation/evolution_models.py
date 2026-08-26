"""Typed ontology evolution snapshots and reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from legal_rag.ontology_validation.evolution_config import EVOLUTION_REPORT_SCHEMA_VERSION

TermKind = Literal["class", "datatype_property", "object_property"]


@dataclass(frozen=True, slots=True)
class OntologyTermSnapshot:
    """One declared local ontology term."""

    iri: str
    kinds: tuple[TermKind, ...]
    deprecated: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable term snapshot."""
        return {"iri": self.iri, "kinds": list(self.kinds), "deprecated": self.deprecated}


@dataclass(frozen=True, slots=True)
class OntologyTermKindChange:
    """One local term whose OWL declaration kind changed."""

    iri: str
    before_kinds: tuple[TermKind, ...]
    after_kinds: tuple[TermKind, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable kind change."""
        return {
            "iri": self.iri,
            "before_kinds": list(self.before_kinds),
            "after_kinds": list(self.after_kinds),
        }


@dataclass(frozen=True, slots=True)
class OntologySemanticHashChange:
    """One retained term or shape whose formal semantic closure changed."""

    iri: str
    before_sha256: str
    after_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable semantic hash change."""
        return {
            "iri": self.iri,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


@dataclass(frozen=True, slots=True)
class OntologyReplacement:
    """One explicit replacement link in the newer ontology."""

    source_iri: str
    replacement_iri: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable replacement link."""
        return {"source_iri": self.source_iri, "replacement_iri": self.replacement_iri}


@dataclass(frozen=True, slots=True)
class OntologyEvolutionIssue:
    """One breaking ontology evolution finding."""

    check_id: str
    subject: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable breaking finding."""
        return {"check_id": self.check_id, "subject": self.subject, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class OntologyEvolutionReport:
    """One semantic diff and breaking-change gate result."""

    checked_at: datetime
    before_project_root: str
    after_project_root: str
    policy_artifact_id: str
    policy_artifact_version: str
    overall_status: Literal["passed", "failed"]
    added_terms: tuple[OntologyTermSnapshot, ...]
    removed_terms: tuple[OntologyTermSnapshot, ...]
    kind_changes: tuple[OntologyTermKindChange, ...]
    changed_term_semantics: tuple[OntologySemanticHashChange, ...]
    added_node_shape_iris: tuple[str, ...]
    removed_node_shape_iris: tuple[str, ...]
    changed_node_shape_constraints: tuple[OntologySemanticHashChange, ...]
    added_mandatory_cq_ids: tuple[str, ...]
    removed_mandatory_cq_ids: tuple[str, ...]
    newly_deprecated_term_iris: tuple[str, ...]
    replacements: tuple[OntologyReplacement, ...]
    issues: tuple[OntologyEvolutionIssue, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable evolution report schema."""
        return {
            "schema_version": EVOLUTION_REPORT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "before_project_root": self.before_project_root,
            "after_project_root": self.after_project_root,
            "policy_artifact_id": self.policy_artifact_id,
            "policy_artifact_version": self.policy_artifact_version,
            "overall_status": self.overall_status,
            "added_terms": [item.to_dict() for item in self.added_terms],
            "removed_terms": [item.to_dict() for item in self.removed_terms],
            "kind_changes": [item.to_dict() for item in self.kind_changes],
            "changed_term_semantics": [item.to_dict() for item in self.changed_term_semantics],
            "added_node_shape_iris": list(self.added_node_shape_iris),
            "removed_node_shape_iris": list(self.removed_node_shape_iris),
            "changed_node_shape_constraints": [item.to_dict() for item in self.changed_node_shape_constraints],
            "added_mandatory_cq_ids": list(self.added_mandatory_cq_ids),
            "removed_mandatory_cq_ids": list(self.removed_mandatory_cq_ids),
            "newly_deprecated_term_iris": list(self.newly_deprecated_term_iris),
            "replacements": [item.to_dict() for item in self.replacements],
            "issues": [item.to_dict() for item in self.issues],
        }
