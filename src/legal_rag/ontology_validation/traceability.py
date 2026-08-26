"""Validate traceability identifiers across ontology artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rdflib import RDF, SH, Graph, Literal, URIRef

from legal_rag.ontology_validation.structure import local_terms

SOURCE_PROVENANCE = "https://w3id.org/legal-ontology-rag/ontology/source-provenance#"
REFERENCE_PREDICATES = {
    "requirement": URIRef(f"{SOURCE_PROVENANCE}requirementReference"),
    "cq": URIRef(f"{SOURCE_PROVENANCE}competencyQuestionReference"),
    "glossary": URIRef(f"{SOURCE_PROVENANCE}glossaryReference"),
    "decision": URIRef(f"{SOURCE_PROVENANCE}modelingDecisionReference"),
}
PATTERN_KEYS = {
    "id",
    "purpose",
    "cq_ids",
    "ontology_term_iris",
    "shape_iris",
    "positive_fixture_ids",
    "boundary_fixture_ids",
}


def collect_ids(value: Any) -> frozenset[str]:
    """Collect stable IDs recursively from a loaded YAML artifact."""
    identifiers: set[str] = set()
    if isinstance(value, Mapping):
        identifier = value.get("id")
        if isinstance(identifier, str):
            identifiers.add(identifier)
        for child in value.values():
            identifiers.update(collect_ids(child))
    elif isinstance(value, list):
        for child in value:
            identifiers.update(collect_ids(child))
    return frozenset(identifiers)


def referenced_ids(graph: Graph, predicate: URIRef) -> frozenset[str]:
    """Return literal IDs referenced through one traceability predicate."""
    return frozenset(str(value) for value in graph.objects(predicate=predicate) if isinstance(value, Literal))


def terms_without_traceability(graph: Graph, namespaces: tuple[str, ...]) -> frozenset[URIRef]:
    """Return local terms that have no registered traceability reference."""
    predicates = tuple(REFERENCE_PREDICATES.values())
    return frozenset(
        term
        for term in local_terms(graph, namespaces)
        if not any(graph.objects(term, predicate) for predicate in predicates)
    )


def shapes_without_traceability(graph: Graph) -> frozenset[URIRef]:
    """Return named node shapes without requirement, CQ, or decision references."""
    required = (
        REFERENCE_PREDICATES["requirement"],
        REFERENCE_PREDICATES["cq"],
        REFERENCE_PREDICATES["decision"],
    )
    return frozenset(
        shape
        for shape in graph.subjects(RDF.type, SH.NodeShape)
        if isinstance(shape, URIRef) and not all(any(graph.objects(shape, predicate)) for predicate in required)
    )


def subjects_for_cq(graphs: Iterable[Graph], cq_id: str) -> frozenset[URIRef]:
    """Return named ontology or shape resources traced to one CQ."""
    predicate = REFERENCE_PREDICATES["cq"]
    return frozenset(
        subject
        for graph in graphs
        for subject in graph.subjects(predicate, Literal(cq_id))
        if isinstance(subject, URIRef)
    )


def ontology_pattern_issues(
    pattern_document: Mapping[str, Any],
    cq_document: Mapping[str, Any],
    fixture_document: Mapping[str, Any],
    ontology_graph: Graph,
    shapes_graph: Graph,
) -> tuple[str, ...]:
    """Validate the minimal pattern manifest and all of its references."""
    issues = []
    if set(pattern_document) != {"schema_version", "artifact_id", "artifact_version", "patterns"}:
        issues.append("pattern-manifest-schema")
    patterns = pattern_document.get("patterns", [])
    pattern_ids = [item.get("id") for item in patterns if isinstance(item, Mapping)]
    if len(pattern_ids) != len(set(pattern_ids)):
        issues.append("duplicate-pattern-id")

    cq_items = cq_document["competency_questions"]
    known_cqs = {item["id"] for item in cq_items}
    mandatory_cqs = {item["id"] for item in cq_items if item["priority"] == "must"}
    fixture_ids = collect_ids(fixture_document)
    ontology_iris = {str(subject) for subject in ontology_graph.subjects() if isinstance(subject, URIRef)}
    shape_iris = {
        str(subject) for subject in shapes_graph.subjects(RDF.type, SH.NodeShape) if isinstance(subject, URIRef)
    }
    covered_cqs = set()
    for pattern in patterns:
        pattern_id = pattern.get("id", "unknown")
        if set(pattern) != PATTERN_KEYS:
            issues.append(f"pattern-schema:{pattern_id}")
            continue
        cq_ids = set(pattern["cq_ids"])
        positive_ids = set(pattern["positive_fixture_ids"])
        boundary_ids = set(pattern["boundary_fixture_ids"])
        if not all(
            (
                pattern["purpose"],
                cq_ids,
                pattern["ontology_term_iris"],
                pattern["shape_iris"],
                positive_ids,
                boundary_ids,
            )
        ):
            issues.append(f"incomplete-pattern:{pattern_id}")
        covered_cqs.update(cq_ids)
        issues.extend(f"unknown-pattern-cq:{pattern_id}:{item}" for item in sorted(cq_ids - known_cqs))
        issues.extend(
            f"unknown-pattern-term:{pattern_id}:{item}"
            for item in sorted(set(pattern["ontology_term_iris"]) - ontology_iris)
        )
        issues.extend(
            f"unknown-pattern-shape:{pattern_id}:{item}" for item in sorted(set(pattern["shape_iris"]) - shape_iris)
        )
        issues.extend(
            f"unknown-pattern-fixture:{pattern_id}:{item}"
            for item in sorted((positive_ids | boundary_ids) - fixture_ids)
        )
    issues.extend(f"cq-without-pattern:{item}" for item in sorted(mandatory_cqs - covered_cqs))
    return tuple(sorted(issues))
