"""Tests for ontology syntax, modules, imports, and modeling guardrails."""

from __future__ import annotations

from rdflib import OWL, RDF, RDFS, Graph, URIRef

from legal_rag.ontology_validation.loader import load_graph
from legal_rag.ontology_validation.models import ModulePolicy
from legal_rag.ontology_validation.paths import OntologyPaths
from legal_rag.ontology_validation.structure import (
    catalog_mappings,
    duplicate_module_ids,
    forbidden_constructs,
    ontology_imports,
    undocumented_terms,
    validate_import_dag,
)


def test_every_turtle_artifact_parses(ontology_paths: OntologyPaths) -> None:
    """Every committed ontology, shape, and fixture is valid Turtle."""
    turtle_paths = tuple(ontology_paths.ontology_dir.rglob("*.ttl"))

    assert turtle_paths
    for path in turtle_paths:
        load_graph(path)


def test_module_policy_matches_ontology_metadata(module_policy: tuple[ModulePolicy, ...]) -> None:
    """Module files, ontology IRIs, version IRIs, and imports match policy."""
    by_id = {module.module_id: module for module in module_policy}

    assert not duplicate_module_ids(module_policy)
    for module in module_policy:
        graph = load_graph(module.path)
        ontology = URIRef(module.ontology_iri)
        expected_imports = frozenset(by_id[module_id].ontology_iri for module_id in module.imports)

        assert module.path.is_file()
        assert (ontology, RDF.type, OWL.Ontology) in graph
        assert (ontology, OWL.versionIRI, URIRef(module.version_iri)) in graph
        assert ontology_imports(graph, module.ontology_iri) == expected_imports


def test_local_import_graph_is_acyclic(module_policy: tuple[ModulePolicy, ...]) -> None:
    """The root and active module dependency graph is acyclic."""
    assert validate_import_dag(module_policy)


def test_catalog_resolves_all_versioned_ontology_iris(
    ontology_paths: OntologyPaths,
    module_policy: tuple[ModulePolicy, ...],
) -> None:
    """The local catalog resolves canonical and versioned ontology IRIs."""
    mappings = catalog_mappings(ontology_paths.catalog)
    expected = {iri: module.path for module in module_policy for iri in (module.ontology_iri, module.version_iri)}

    for iri, path in expected.items():
        assert mappings[iri].resolve() == path.resolve()
        assert mappings[iri].is_file()
    assert all(path.is_file() for path in mappings.values())


def test_deferred_owl_and_swrl_constructs_are_absent(ontology_graph: Graph) -> None:
    """V0.1 avoids unapproved strong OWL axioms and executable rule semantics."""
    assert not forbidden_constructs(ontology_graph)


def test_local_terms_have_traceable_human_documentation(
    ontology_graph: Graph,
    module_policy: tuple[ModulePolicy, ...],
) -> None:
    """Local classes, properties, and controlled values have readable documentation."""
    namespaces = tuple(module.namespace for module in module_policy)

    assert not undocumented_terms(ontology_graph, namespaces)


def test_valid_during_has_no_domain_that_forces_accidental_classification(ontology_graph: Graph) -> None:
    """The shared validity property does not infer an artificial common domain class."""
    valid_during = URIRef("https://w3id.org/legal-ontology-rag/ontology/temporal-change#validDuring")

    assert not tuple(ontology_graph.objects(valid_during, RDFS.domain))


def test_legal_change_rule_ranges_are_supplied_by_normative_module(ontology_graph: Graph) -> None:
    """Cross-module ranges constrain legal-change links to penalty-rule versions."""
    temporal = "https://w3id.org/legal-ontology-rag/ontology/temporal-change#"
    penalty_rule = URIRef("https://w3id.org/legal-ontology-rag/ontology/normative-rule#PenaltyRule")

    assert (URIRef(f"{temporal}changesEarlierRule"), RDFS.range, penalty_rule) in ontology_graph
    assert (URIRef(f"{temporal}producesLaterRule"), RDFS.range, penalty_rule) in ontology_graph
