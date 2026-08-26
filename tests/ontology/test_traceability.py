"""Tests for ontology-to-requirement traceability."""

from __future__ import annotations

from rdflib import Graph

from legal_rag.ontology_validation.loader import load_graph, load_query_cases, load_yaml
from legal_rag.ontology_validation.models import ModulePolicy
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.traceability import (
    REFERENCE_PREDICATES,
    collect_ids,
    ontology_pattern_issues,
    referenced_ids,
    shapes_without_traceability,
    subjects_for_cq,
    terms_without_traceability,
)


def test_all_traceability_references_resolve_to_versioned_registries(
    ontology_paths: OntologyPaths,
    ontology_graph: Graph,
) -> None:
    """Every traceability literal resolves to a known registry identifier."""
    shapes_graph = load_graph(ontology_paths.shapes)
    combined = ontology_graph + shapes_graph
    registries = {
        "requirement": collect_ids(load_yaml(ontology_paths.requirements)),
        "cq": collect_ids(load_yaml(ontology_paths.competency_questions)),
        "glossary": collect_ids(load_yaml(ontology_paths.glossary)),
        "decision": collect_ids(load_yaml(ontology_paths.modeling_decisions)),
    }

    for kind, predicate in REFERENCE_PREDICATES.items():
        assert referenced_ids(combined, predicate) <= registries[kind]


def test_requirement_registry_uses_minimal_schema(ontology_paths: OntologyPaths) -> None:
    """Requirements retain only versioning, grouping, identity, and statements."""
    document = load_yaml(ontology_paths.requirements)

    assert set(document) == {
        "schema_version",
        "artifact_id",
        "artifact_version",
        "status",
        "requirement_categories",
    }
    assert [category["id"] for category in document["requirement_categories"]] == [
        "purpose",
        "intended_use",
        "in_scope",
        "required_granularity",
    ]
    for category in document["requirement_categories"]:
        assert set(category) == {"id", "requirements"}
        assert all(set(requirement) == {"id", "statement"} for requirement in category["requirements"])


def test_all_local_terms_have_traceability(
    ontology_graph: Graph,
    module_policy: tuple[ModulePolicy, ...],
) -> None:
    """Every local OWL or controlled term traces to a reviewed artifact."""
    namespaces = tuple(module.namespace for module in module_policy)

    assert not terms_without_traceability(ontology_graph, namespaces)


def test_all_named_node_shapes_have_traceability(ontology_paths: OntologyPaths) -> None:
    """Every named node shape traces to requirements, CQs, and decisions."""
    assert not shapes_without_traceability(load_graph(ontology_paths.shapes))


def test_every_mandatory_cq_has_formal_and_executable_coverage(
    ontology_paths: OntologyPaths,
    ontology_graph: Graph,
) -> None:
    """Each mandatory CQ maps to ontology terms or shapes and an executable fixture query."""
    cq_document = load_yaml(ontology_paths.competency_questions)
    mandatory_cqs = {item["id"] for item in cq_document["competency_questions"] if item["priority"] == "must"}
    shapes_graph = load_graph(ontology_paths.shapes)
    query_cases = {
        case.cq_id: case for case in load_query_cases(ontology_paths.project_root, ontology_paths.query_manifest)
    }

    for cq_id in mandatory_cqs:
        assert subjects_for_cq((ontology_graph, shapes_graph), cq_id)
        assert query_cases[cq_id].query_path.is_file()
        assert query_cases[cq_id].fixture_path.is_file()


def test_ontology_patterns_are_minimal_complete_and_reference_existing_artifacts(
    ontology_paths: OntologyPaths,
    ontology_graph: Graph,
) -> None:
    """Every mandatory CQ is implemented through a tested reusable ontology pattern."""
    issues = ontology_pattern_issues(
        load_yaml(ontology_paths.patterns),
        load_yaml(ontology_paths.competency_questions),
        load_yaml(ontology_paths.pilot_fixtures),
        ontology_graph,
        load_graph(ontology_paths.shapes),
    )

    assert not issues
