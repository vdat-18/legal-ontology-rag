"""Tests for the custom traffic-penalty entity/edge types used by Graphiti extraction."""

from __future__ import annotations

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from legal_rag.graphiti_rag.ontology import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES


def test_entity_types_have_expected_keys() -> None:
    """The entity ontology defines exactly the planned traffic-penalty node types."""
    assert set(ENTITY_TYPES) == {
        "Violation",
        "Penalty",
        "RemedialMeasure",
        "LegalDocument",
        "Provision",
        "Subject",
        "Authority",
    }


def test_edge_types_have_expected_keys() -> None:
    """The edge ontology defines exactly the planned traffic-penalty relationship types."""
    assert set(EDGE_TYPES) == {
        "PENALIZES",
        "APPLIES_TO",
        "REQUIRES_REMEDY",
        "DEFINED_IN",
        "PART_OF",
        "ENFORCED_BY",
        "AMENDS",
    }


def test_entity_type_fields_do_not_collide_with_base_entity_node_fields() -> None:
    """Custom entity type fields must not shadow Graphiti's base EntityNode fields."""
    base_fields = set(EntityNode.model_fields)

    for name, model in ENTITY_TYPES.items():
        overlap = set(model.model_fields) & base_fields
        assert not overlap, f"{name} has fields colliding with EntityNode: {overlap}"


def test_edge_type_fields_do_not_collide_with_base_entity_edge_fields() -> None:
    """Custom edge type fields must not shadow Graphiti's base EntityEdge fields."""
    base_fields = set(EntityEdge.model_fields)

    for name, model in EDGE_TYPES.items():
        overlap = set(model.model_fields) & base_fields
        assert not overlap, f"{name} has fields colliding with EntityEdge: {overlap}"


def test_edge_type_map_only_references_declared_types() -> None:
    """Every constrained pair uses declared entity and edge type names, so no rule is a no-op."""
    for (source, target), edge_names in EDGE_TYPE_MAP.items():
        assert source in ENTITY_TYPES, source
        assert target in ENTITY_TYPES, target
        for edge_name in edge_names:
            assert edge_name in EDGE_TYPES, edge_name


def test_every_edge_type_has_at_least_one_allowed_pair() -> None:
    """An edge type absent from the map could never be extracted, so it must not be declared."""
    constrained = {name for names in EDGE_TYPE_MAP.values() for name in names}

    assert constrained == set(EDGE_TYPES)


def test_edge_type_map_does_not_allow_untyped_pairs() -> None:
    """('Entity', 'Entity') would re-enable every edge type between any two nodes, voiding the map."""
    assert ("Entity", "Entity") not in EDGE_TYPE_MAP


def test_penalizes_only_runs_from_a_violation_to_a_penalty() -> None:
    """The reversed and same-type edges seen in the built graph are no longer permitted."""
    allowed = {pair for pair, names in EDGE_TYPE_MAP.items() if "PENALIZES" in names}

    assert allowed == {("Violation", "Penalty")}
