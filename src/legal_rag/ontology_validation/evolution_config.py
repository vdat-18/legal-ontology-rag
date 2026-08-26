"""Typed configuration for ontology evolution gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

EVOLUTION_POLICY_SCHEMA_VERSION = "0.1.0"
EVOLUTION_REPORT_SCHEMA_VERSION = "0.2.0"

_BREAKING_CHECKS = {
    "changed_term_kind",
    "changed_active_term_semantics",
    "removed_active_term",
    "removed_mandatory_cq",
    "removed_node_shape",
    "changed_node_shape_constraints",
}
_CHANGE_EVIDENCE = {
    "accepted_requirement_or_cq",
    "modeling_decision",
    "passing_semantic_qa",
    "project_owner_acceptance",
}


@dataclass(frozen=True, slots=True)
class OntologyEvolutionSettings:
    """Versioned lifecycle boundaries and semantic breaking-change checks."""

    artifact_id: str
    artifact_version: str
    new_data_action: str
    llm_tbox_mutation: str
    required_change_evidence: tuple[str, ...]
    breaking_checks: frozenset[str]
    deprecation_predicate_iri: str
    replacement_predicate_iri: str
    require_retained_declaration: bool
    meaning_change_action: str

    @classmethod
    def from_path(cls, path: Path) -> OntologyEvolutionSettings:
        """Load and validate one versioned ontology-evolution policy."""
        document = load_yaml(path)
        _reject_unknown_fields(
            document,
            {
                "schema_version",
                "artifact_id",
                "artifact_version",
                "status",
                "lifecycle",
                "breaking_changes",
                "deprecation",
                "semantic_boundaries",
            },
            "evolution policy",
        )
        if document["schema_version"] != EVOLUTION_POLICY_SCHEMA_VERSION:
            msg = f"Unsupported ontology-evolution policy: {document['schema_version']}"
            raise ValueError(msg)
        lifecycle = document["lifecycle"]
        breaking_changes = document["breaking_changes"]
        deprecation = document["deprecation"]
        _reject_unknown_fields(
            lifecycle,
            {"new_data_action", "llm_tbox_mutation", "required_change_evidence"},
            "evolution lifecycle",
        )
        _reject_unknown_fields(breaking_changes, {"fail_on"}, "evolution breaking-change policy")
        _reject_unknown_fields(
            deprecation,
            {
                "predicate_iri",
                "replacement_predicate_iri",
                "require_retained_declaration",
                "meaning_change_action",
            },
            "evolution deprecation policy",
        )
        required_evidence = tuple(lifecycle["required_change_evidence"])
        breaking_checks = frozenset(breaking_changes["fail_on"])
        if set(required_evidence) != _CHANGE_EVIDENCE:
            msg = "Ontology evolution requires the complete reviewed change-evidence set"
            raise ValueError(msg)
        unknown_checks = breaking_checks - _BREAKING_CHECKS
        if unknown_checks:
            msg = f"Unsupported ontology breaking-change check: {sorted(unknown_checks)[0]}"
            raise ValueError(msg)
        if lifecycle["new_data_action"] != "populate_instances":
            msg = "New ontology data must default to instance population"
            raise ValueError(msg)
        if lifecycle["llm_tbox_mutation"] != "prohibited":
            msg = "LLM TBox mutation must remain prohibited"
            raise ValueError(msg)
        if deprecation["meaning_change_action"] != "mint_new_iri":
            msg = "Changed ontology meaning must mint a new IRI"
            raise ValueError(msg)
        return cls(
            artifact_id=document["artifact_id"],
            artifact_version=document["artifact_version"],
            new_data_action=lifecycle["new_data_action"],
            llm_tbox_mutation=lifecycle["llm_tbox_mutation"],
            required_change_evidence=required_evidence,
            breaking_checks=breaking_checks,
            deprecation_predicate_iri=deprecation["predicate_iri"],
            replacement_predicate_iri=deprecation["replacement_predicate_iri"],
            require_retained_declaration=deprecation["require_retained_declaration"],
            meaning_change_action=deprecation["meaning_change_action"],
        )


def _reject_unknown_fields(payload: dict[str, object], allowed: set[str], artifact: str) -> None:
    """Reject policy fields without runtime or governance semantics."""
    unknown = set(payload) - allowed
    if unknown:
        msg = f"Unknown {artifact} fields: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
