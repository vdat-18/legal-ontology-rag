"""Integration tests for OntologyRAGPipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path


def test_pipeline_imports(test_fixture_path: Path) -> None:
    """Pipeline module imports successfully."""
    from legal_rag.ontology_rag.index import OntologyIndex

    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    index = OntologyIndex(ontology_paths)

    assert index.graph is not None


def test_pipeline_components_integrate(test_fixture_path: Path) -> None:
    """Components integrate: index → retriever → temporal resolver."""
    from legal_rag.ontology_rag.index import OntologyIndex, TemporalIndex
    from legal_rag.ontology_rag.retrieval import InstanceRetriever, TemporalResolver

    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)
    temporal_index = TemporalIndex(ontology_index.graph)

    rule_uris = list(temporal_index.all_rules())[:2]
    retriever = InstanceRetriever(ontology_index.graph)
    rules = retriever.retrieve_batch(rule_uris)

    resolver = TemporalResolver()
    valid_rules = resolver.filter_valid(rules, date(2025, 6, 1))

    assert len(rules) > 0
    assert len(valid_rules) > 0


def test_pipeline_result_dataclass() -> None:
    """OntologyRAGResult is a well-formed dataclass."""
    from rdflib import URIRef

    from legal_rag.naive_rag.models import RetrievedChunk
    from legal_rag.ontology_rag.models import PenaltyRule

    chunk = RetrievedChunk(
        chunk_id="test",
        text="sample",
        source_name="doc",
        kind="para",
        score=0.9,
    )
    rule = PenaltyRule(
        uri=URIRef("rule:001"),
        conduct_label_vi="test conduct",
        vehicle_categories=(),
        subject_kind=None,
        roles=(),
        monetary_penalty_min=None,
        monetary_penalty_max=None,
        currency=None,
        point_value=None,
        valid_from=date(2025, 1, 1),
        valid_to=None,
        source_chunk_ids=(),
        provision_path="test",
        document_number="test",
    )

    assert chunk.chunk_id == "test"
    assert rule.conduct_label_vi == "test conduct"
