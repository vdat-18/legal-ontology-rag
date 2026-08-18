"""Typed configuration for external ontology reuse review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

REUSE_REGISTRY_SCHEMA_VERSION = "0.1.0"
REUSE_REPORT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ReusedTerm:
    """One external term intentionally used by a formal artifact."""

    iri: str
    local_use: str
    declaration_in_owl_closure: bool


@dataclass(frozen=True, slots=True)
class ExternalReuseResource:
    """One reviewed external vocabulary, standard, ontology, or pattern."""

    decision_id: str
    resource: str
    authority: str
    source_version: str
    source_url: str
    namespace: str | None
    license: str
    attribution: str
    decision: str
    policy_mode: str
    formal_use: str
    owl_import: bool
    review_status: str
    compatibility: str
    boundaries: tuple[str, ...]
    terms: tuple[ReusedTerm, ...]


@dataclass(frozen=True, slots=True)
class ExternalReuseSettings:
    """Versioned reuse-review scope, decisions, evidence, and approval state."""

    artifact_id: str
    artifact_version: str
    project_owner_approval: str
    formal_paths: tuple[Path, ...]
    foundational_namespaces: tuple[str, ...]
    resources: tuple[ExternalReuseResource, ...]

    @classmethod
    def from_path(cls, path: Path, project_root: Path) -> ExternalReuseSettings:
        """Load the versioned external-reuse registry."""
        document = load_yaml(path)
        if document["schema_version"] != REUSE_REGISTRY_SCHEMA_VERSION:
            msg = f"Unsupported external-reuse registry: {document['schema_version']}"
            raise ValueError(msg)
        return cls(
            artifact_id=document["artifact_id"],
            artifact_version=document["artifact_version"],
            project_owner_approval=document["review"]["project_owner_approval"],
            formal_paths=tuple(project_root / item for item in document["scope"]["formal_artifacts"]),
            foundational_namespaces=tuple(document["scope"]["foundational_namespaces"]),
            resources=tuple(
                ExternalReuseResource(
                    decision_id=item["id"],
                    resource=item["resource"],
                    authority=item["authority"],
                    source_version=item["source_version"],
                    source_url=item["source_url"],
                    namespace=item["namespace"],
                    license=item["license"],
                    attribution=item["attribution"],
                    decision=item["decision"],
                    policy_mode=item["policy_mode"],
                    formal_use=item["formal_use"],
                    owl_import=item["owl_import"],
                    review_status=item["review_status"],
                    compatibility=item["compatibility"],
                    boundaries=tuple(item["boundaries"]),
                    terms=tuple(
                        ReusedTerm(term["iri"], term["local_use"], term["declaration_in_owl_closure"])
                        for term in item["terms"]
                    ),
                )
                for item in document["resources"]
            ),
        )
