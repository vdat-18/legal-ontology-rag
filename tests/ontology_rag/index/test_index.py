"""Tests for ontology index modules."""

from __future__ import annotations

from pathlib import Path

from legal_rag.ontology_rag.index import OntologyIndex, TemporalIndex


def test_ontology_index_loads_ttl(test_fixture_path: Path) -> None:
    """OntologyIndex loads TTL files."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    index = OntologyIndex(ontology_paths)
    assert len(index.graph) > 0
    namespaces = dict(index.graph.namespaces())
    assert "lornr" in namespaces


def test_temporal_index_extracts_periods(test_fixture_path: Path) -> None:
    """TemporalIndex extracts valid periods from rules."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)
    temporal_index = TemporalIndex(ontology_index.graph)
    rules = temporal_index.all_rules()
    assert len(rules) > 0
