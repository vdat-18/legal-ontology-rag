"""Tests for ontology index modules."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from rdflib import DCTERMS, RDF, TIME, XSD, Graph, Literal, Namespace, URIRef

from legal_rag.ontology_rag.index import ChunkIndex, OntologyIndex, TemporalIndex

LORNR = Namespace("https://w3id.org/legal-ontology-rag/ontology/normative-rule#")
LORSP = Namespace("https://w3id.org/legal-ontology-rag/ontology/source-provenance#")
LORTC = Namespace("https://w3id.org/legal-ontology-rag/ontology/temporal-change#")
EX = Namespace("https://example.test/")


def test_ontology_index_loads_ttl(test_fixture_path: Path) -> None:
    """OntologyIndex loads TTL files."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    index = OntologyIndex(ontology_paths)
    assert len(index.graph) > 0
    namespaces = dict(index.graph.namespaces())
    assert "lornr" in namespaces


def test_ontology_index_uses_a_valid_file_base_uri(tmp_path: Path) -> None:
    """Relative RDF identifiers resolve against a canonical file URI on Windows."""
    ontology_path = tmp_path / "relative.ttl"
    ontology_path.write_text('@prefix ex: <https://example.test/> .\n<rule> ex:value "value" .\n', encoding="utf-8")

    index = OntologyIndex([ontology_path])

    expected_subject = URIRef((tmp_path / "rule").resolve().as_uri())
    assert (expected_subject, EX.value, Literal("value")) in index.graph


def test_ontology_index_loads_uppercase_turtle_extension(tmp_path: Path) -> None:
    """Runtime loading accepts the same case-insensitive Turtle suffix as CLI validation."""
    ontology_path = tmp_path / "release.TTL"
    ontology_path.write_text('@prefix ex: <https://example.test/> .\nex:rule ex:value "value" .\n', encoding="utf-8")

    index = OntologyIndex([ontology_path])

    assert (EX.rule, EX.value, Literal("value")) in index.graph


def test_ontology_index_rejects_missing_input(tmp_path: Path) -> None:
    """A configured ontology path cannot be silently omitted from the active graph."""
    missing_path = tmp_path / "missing.ttl"

    with pytest.raises(FileNotFoundError, match="Ontology file does not exist"):
        OntologyIndex([missing_path])


def test_ontology_index_rejects_unsupported_input(tmp_path: Path) -> None:
    """Only Turtle inputs accepted by the evaluation CLI can enter the active graph."""
    ontology_path = tmp_path / "release.rdf"
    ontology_path.write_text("<rdf:RDF />", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be a \.ttl file"):
        OntologyIndex([ontology_path])


def test_ontology_index_rejects_an_empty_path_set() -> None:
    """An empty ontology configuration cannot produce an apparently initialized index."""
    with pytest.raises(ValueError, match="At least one ontology Turtle file"):
        OntologyIndex([])


def test_temporal_index_extracts_periods(test_fixture_path: Path) -> None:
    """TemporalIndex extracts valid periods from rules."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)
    temporal_index = TemporalIndex(ontology_index.graph)
    rules = temporal_index.all_rules()
    assert len(rules) > 0


def test_chunk_index_maps_component_evidence_to_explicit_penalty_rules() -> None:
    """Component evidence resolves to its typed owner without becoming a rule."""
    graph = Graph()
    graph.add((EX.chunk, DCTERMS.identifier, Literal("chunk-1")))
    graph.add((EX.component_chunk, DCTERMS.identifier, Literal("chunk-2")))
    graph.add((EX.condition_chunk, DCTERMS.identifier, Literal("chunk-3")))
    graph.add((EX.actual, RDF.type, LORNR.PenaltyRule))
    graph.add((EX.actual, LORSP.sourcedFromChunk, EX.chunk))
    graph.add((EX.actual, LORNR.hasApplicability, EX.component))
    graph.add((EX.component, LORSP.sourcedFromChunk, EX.component_chunk))
    graph.add((EX.component, LORNR.hasConditionExpression, EX.condition_root))
    graph.add((EX.condition_root, LORNR.hasOperand, EX.condition_leaf))
    graph.add((EX.condition_leaf, LORSP.sourcedFromChunk, EX.condition_chunk))
    graph.add((URIRef("https://example.test/rule/component"), LORSP.sourcedFromChunk, EX.component_chunk))

    index = ChunkIndex(graph)

    assert index.get_instances("chunk-1") == [EX.actual]
    assert index.get_instances("chunk-2") == [EX.actual]
    assert index.get_instances("chunk-3") == [EX.actual]


def test_temporal_index_partitions_half_open_and_unknown_intervals() -> None:
    """The end boundary is exclusive and missing bounds stay indeterminate."""
    graph = Graph()
    _add_interval(graph, EX.old, date(2025, 1, 1), date(2026, 8, 15))
    _add_interval(graph, EX.new, date(2026, 8, 15), date(2027, 1, 1))
    _add_interval(graph, EX.unknown_end, date(2025, 1, 1), None)
    _add_interval(graph, EX.unknown_start, None, date(2027, 1, 1))
    graph.add((EX.no_interval, RDF.type, LORNR.PenaltyRule))
    index = TemporalIndex(graph)

    resolution = index.partition(
        [EX.old, EX.new, EX.unknown_end, EX.unknown_start, EX.no_interval],
        date(2026, 8, 15),
    )

    assert resolution.valid == (EX.new,)
    assert resolution.invalid == (EX.old,)
    assert resolution.indeterminate == (EX.unknown_end, EX.unknown_start, EX.no_interval)
    assert index.filter_valid(list(index.all_rules()), date(2026, 8, 15)) == [EX.new]


def _add_interval(graph: Graph, rule: URIRef, valid_from: date | None, valid_to: date | None) -> None:
    """Add one test penalty rule and its optional temporal bounds."""
    interval = URIRef(f"{rule}/validity")
    graph.add((rule, RDF.type, LORNR.PenaltyRule))
    graph.add((rule, LORTC.validDuring, interval))
    if valid_from is not None:
        beginning = URIRef(f"{interval}/beginning")
        graph.add((interval, TIME.hasBeginning, beginning))
        graph.add(
            (
                beginning,
                TIME.inXSDDateTime,
                Literal(f"{valid_from.isoformat()}T00:00:00Z", datatype=XSD.dateTime),
            )
        )
    if valid_to is not None:
        end = URIRef(f"{interval}/end")
        graph.add((interval, TIME.hasEnd, end))
        graph.add(
            (
                end,
                TIME.inXSDDateTime,
                Literal(f"{valid_to.isoformat()}T00:00:00Z", datatype=XSD.dateTime),
            )
        )
