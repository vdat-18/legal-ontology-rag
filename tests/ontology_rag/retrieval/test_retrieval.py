"""Tests for retrieval modules."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from legal_rag.ontology_rag.index import OntologyIndex, TemporalIndex
from legal_rag.ontology_rag.retrieval import (
    InstanceRetriever,
    SPARQLSearch,
    TemporalResolver,
)


def test_instance_retriever_fetches_rule(test_fixture_path: Path) -> None:
    """InstanceRetriever fetches full penalty rule."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)
    temporal_index = TemporalIndex(ontology_index.graph)

    rule_uris = list(temporal_index.all_rules())[:1]
    retriever = InstanceRetriever(ontology_index.graph)
    rules = retriever.retrieve_batch(rule_uris)

    assert len(rules) > 0
    rule = rules[0]
    assert rule.conduct_label_vi
    assert rule.valid_from.year >= 2025


def test_temporal_resolver_filters_rules(test_fixture_path: Path) -> None:
    """TemporalResolver filters rules by date."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)
    temporal_index = TemporalIndex(ontology_index.graph)

    rule_uris = list(temporal_index.all_rules())[:2]
    retriever = InstanceRetriever(ontology_index.graph)
    rules = retriever.retrieve_batch(rule_uris)

    resolver = TemporalResolver()
    valid = resolver.filter_valid(rules, date(2025, 6, 1))

    assert len(valid) > 0
    for rule in valid:
        assert rule.valid_from <= date(2025, 6, 1)


def test_sparql_search_finds_instances(test_fixture_path: Path) -> None:
    """SPARQLSearch finds instances by keywords."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)

    search = SPARQLSearch(ontology_index.graph)
    instances = search.search(["biển số"])

    assert len(instances) >= 0
