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
