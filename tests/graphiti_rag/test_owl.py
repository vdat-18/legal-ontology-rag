"""Tests for rendering the traffic-penalty ontology as OWL."""

from __future__ import annotations

from legal_rag.graphiti_rag.ontology import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES
from legal_rag.graphiti_rag.owl import build_owl


def test_declares_every_class_and_object_property() -> None:
    """The published ontology covers the whole schema, so nothing enforced is left undocumented."""
    document = build_owl()

    for name in ENTITY_TYPES:
        assert f":{name} a owl:Class ;" in document
    for name in EDGE_TYPES:
        assert f":{name} a owl:ObjectProperty ;" in document


def test_uses_union_for_a_property_allowed_from_several_classes() -> None:
    """Listing classes directly would mean their intersection, requiring both types at once."""
    document = build_owl()

    assert "rdfs:domain [ a owl:Class ; owl:unionOf ( :Provision :Violation ) ] ;" in document


def test_uses_a_plain_class_when_only_one_is_allowed() -> None:
    """A single-class domain needs no union wrapper, keeping the axiom readable."""
    document = build_owl()

    assert "rdfs:domain :Violation ;\n    rdfs:range :Penalty ." in document


def test_domains_and_ranges_come_from_the_enforced_edge_type_map() -> None:
    """Published axioms must match the constraint the indexer applies, not a hand-kept copy."""
    document = build_owl()

    for (source, target), edge_names in EDGE_TYPE_MAP.items():
        for edge_name in edge_names:
            section = document.split(f":{edge_name} a owl:ObjectProperty ;")[1].split(" .\n")[0]
            assert source in section, (edge_name, source)
            assert target in section, (edge_name, target)


def test_does_not_assert_disjointness_the_graph_violates() -> None:
    """Classes are not disjoint, so the ontology stays consistent with its own graph.

    Entities legitimately carry two labels at once (a subject who commits a violation), which
    a disjointness axiom would report as an error.
    """
    assert "owl:disjointWith" not in build_owl()
