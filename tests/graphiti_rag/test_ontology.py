"""Tests for the custom traffic-penalty entity/edge types used by Graphiti extraction."""

from __future__ import annotations

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from legal_rag.graphiti_rag.ontology import EDGE_TYPES, ENTITY_TYPES


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
