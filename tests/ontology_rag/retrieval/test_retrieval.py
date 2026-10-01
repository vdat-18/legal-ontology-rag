"""Tests for retrieval modules."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from rdflib import DCTERMS, RDF, RDFS, Graph, Literal, Namespace, URIRef

from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.ontology_rag.index import OntologyIndex, TemporalIndex
from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
from legal_rag.ontology_rag.retrieval import (
    ChunkLinker,
    InstanceRetriever,
    SPARQLSearch,
    TemporalResolver,
)

LORNR = Namespace("https://w3id.org/legal-ontology-rag/ontology/normative-rule#")
LORSP = Namespace("https://w3id.org/legal-ontology-rag/ontology/source-provenance#")
EX = Namespace("https://example.test/")

_COMPLETE_RULE_TTL = """
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix ex: <https://example.test/> .
@prefix lorlc: <https://w3id.org/legal-ontology-rag/ontology/legal-consequence#> .
@prefix loroc: <https://w3id.org/legal-ontology-rag/ontology/occurrence-context#> .
@prefix lornr: <https://w3id.org/legal-ontology-rag/ontology/normative-rule#> .
@prefix lorsp: <https://w3id.org/legal-ontology-rag/ontology/source-provenance#> .
@prefix lortc: <https://w3id.org/legal-ontology-rag/ontology/temporal-change#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:resource dcterms:identifier "168/2024/NĐ-CP" ; lorsp:hasExpression ex:expression .
ex:expression lorsp:containsProvision ex:rule-provision, ex:component-provision,
    ex:exception-provision, ex:excluded-provision .
ex:rule-provision lorsp:structuralPath "Điều 7 khoản 1" .
ex:component-provision lorsp:structuralPath "Điều 7 khoản 2" .
ex:exception-provision lorsp:structuralPath "Điều 7 khoản 3" .
ex:excluded-provision lorsp:structuralPath "Điều 8 khoản 1" .
ex:chunk-rule dcterms:identifier "chunk-rule" .
ex:chunk-components dcterms:identifier "chunk-components" .
ex:chunk-exception dcterms:identifier "chunk-exception" .

ex:rule a lornr:PenaltyRule ;
    lornr:hasConductSpecification ex:conduct-specification ;
    lornr:hasApplicability ex:applicability ;
    lornr:hasException ex:exception ;
    lorlc:imposesConsequence ex:warning, ex:monetary, ex:points, ex:additional,
        ex:remedy, ex:referenced ;
    lortc:validDuring ex:validity ;
    lorsp:groundedIn ex:rule-provision ;
    lorsp:sourcedFromChunk ex:chunk-rule .
ex:conduct-specification lornr:usesConductConcept ex:conduct ;
    lorsp:groundedIn ex:component-provision ;
    lorsp:sourcedFromChunk ex:chunk-components .
ex:conduct skos:prefLabel "Không chấp hành hiệu lệnh"@vi .
ex:applicability lornr:requiresSubjectKind loroc:naturalPerson ;
    lornr:requiresRole ex:driver-role, ex:owner-role ;
    lornr:requiresVehicleCategory ex:motorcycle, ex:car ;
    lornr:hasConditionExpression ex:condition-all ;
    lorsp:groundedIn ex:component-provision ;
    lorsp:sourcedFromChunk ex:chunk-components .
ex:driver-role skos:prefLabel "Người điều khiển"@vi .
ex:owner-role skos:prefLabel "Chủ phương tiện"@vi .
ex:motorcycle skos:prefLabel "Xe mô tô"@vi .
ex:car skos:prefLabel "Xe ô tô"@vi .

ex:condition-all a lornr:AllOf ; lornr:hasOperand ex:condition-atomic, ex:condition-not ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:condition-atomic a lornr:AtomicCondition ; rdfs:label "Đang tham gia giao thông"@vi ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:condition-not a lornr:Not ; lornr:hasOperand ex:condition-exempt ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:condition-exempt a lornr:AtomicCondition ; rdfs:label "Thuộc trường hợp miễn trừ"@vi ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .

ex:monetary a lorlc:MonetaryPenalty ; lorlc:hasAmountRange ex:amount-range ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:amount-range lorlc:minimumAmount "4000000"^^xsd:decimal ;
    lorlc:maximumAmount "6000000"^^xsd:decimal ; lorlc:currency lorlc:VND .
ex:warning a lorlc:Warning ; rdfs:label "Cảnh cáo"@vi ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:points a lorlc:PointDeduction ; lorlc:pointValue 0 ; lorlc:affectsTarget ex:license ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:additional a lorlc:AdditionalSanction ; lorlc:minimumDurationMonths 1 ;
    lorlc:maximumDurationMonths 3 ; lorlc:affectsTarget ex:vehicle ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:remedy a lorlc:RemedialMeasure ; lorlc:affectsTarget ex:vehicle ;
    lorlc:mustBePerformedBy ex:owner-role ; lorsp:groundedIn ex:component-provision ;
    lorsp:sourcedFromChunk ex:chunk-components .
ex:referenced a lorlc:ReferencedPenalty ; rdfs:label "Áp dụng mức phạt được dẫn chiếu"@vi ;
    lorsp:groundedIn ex:component-provision ; lorsp:sourcedFromChunk ex:chunk-components .
ex:license rdfs:label "Giấy phép lái xe"@vi .
ex:vehicle rdfs:label "Phương tiện"@vi .

ex:exception a lornr:LegalException ; lornr:citesExcludedProvision ex:excluded-provision ;
    lornr:hasProvisionCitation ex:citation ; lornr:redirectsTo ex:specific-rule ;
    lorsp:groundedIn ex:exception-provision ; lorsp:sourcedFromChunk ex:chunk-exception .
ex:citation lorsp:citationText "trừ trường hợp tại Điều 8 khoản 1"@vi ;
    lorsp:refersToProvision ex:excluded-provision .
ex:validity time:hasBeginning ex:beginning ; time:hasEnd ex:end .
ex:beginning time:inXSDDateTime "2025-01-01T00:00:00Z"^^xsd:dateTime .
ex:end time:inXSDDateTime "2026-01-01T00:00:00Z"^^xsd:dateTime .
"""


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
    assert rule.valid_from is not None
    assert rule.valid_from.year >= 2025


def test_instance_retriever_preserves_complete_rule_contract() -> None:
    """InstanceRetriever preserves multi-valued semantics and component provenance."""
    graph = Graph().parse(data=_COMPLETE_RULE_TTL, format="turtle")

    rules = InstanceRetriever(graph).retrieve_batch([EX.rule, EX.rule])

    assert len(rules) == 1
    rule = rules[0]
    assert rule.conduct_label_vi == "Không chấp hành hiệu lệnh"
    assert rule.vehicle_categories == ("Xe ô tô", "Xe mô tô")
    assert rule.subject_kind == "NaturalPerson"
    assert rule.roles == ("Người điều khiển", "Chủ phương tiện")
    assert rule.document_number == "168/2024/NĐ-CP"
    assert rule.provision_path == "Điều 7 khoản 1"
    assert rule.valid_from == date(2025, 1, 1)
    assert rule.valid_to == date(2026, 1, 1)
    assert rule.source_chunk_ids == ("chunk-components", "chunk-exception", "chunk-rule")
    assert rule.rule_provenance.source_chunk_ids == ("chunk-rule",)
    assert rule.conduct_provenance[0].source_chunk_ids == ("chunk-components",)

    assert {item.kind for item in rule.consequences} == {
        "AdditionalSanction",
        "MonetaryPenalty",
        "PointDeduction",
        "ReferencedPenalty",
        "RemedialMeasure",
        "Warning",
    }
    monetary = next(item for item in rule.consequences if item.kind == "MonetaryPenalty")
    assert monetary.minimum_amount == Decimal("4000000")
    assert monetary.maximum_amount == Decimal("6000000")
    assert monetary.currency == "VND"
    assert rule.monetary_penalty_min == Decimal("4000000")
    assert rule.monetary_penalty_max == Decimal("6000000")
    assert rule.currency == "VND"

    points = next(item for item in rule.consequences if item.kind == "PointDeduction")
    assert points.point_value == 0
    assert rule.point_value == 0
    assert points.affected_targets[0].label_vi == "Giấy phép lái xe"
    additional = next(item for item in rule.consequences if item.kind == "AdditionalSanction")
    assert additional.minimum_duration_months == 1
    assert additional.maximum_duration_months == 3
    remedy = next(item for item in rule.consequences if item.kind == "RemedialMeasure")
    assert remedy.responsible_roles[0].label_vi == "Chủ phương tiện"

    condition = rule.condition_expressions[0]
    assert condition.kind == "AllOf"
    assert tuple(item.kind for item in condition.operands) == ("Atomic", "Not")
    assert condition.operands[1].operands[0].label_vi == "Thuộc trường hợp miễn trừ"

    exception = rule.exceptions[0]
    assert exception.excluded_provision_paths == ("Điều 8 khoản 1",)
    assert exception.redirected_rule_uris == (EX["specific-rule"],)
    assert exception.provenance.source_chunk_ids == ("chunk-exception",)
    assert exception.citations[0].citation_text_vi == "trừ trường hợp tại Điều 8 khoản 1"
    assert exception.citations[0].referenced_provision_paths == ("Điều 8 khoản 1",)


def test_instance_retriever_preserves_unknown_valid_from() -> None:
    """A missing validity beginning remains unknown instead of dropping the rule."""
    graph = Graph().parse(
        data="""
        @prefix ex: <https://example.test/> .
        @prefix lornr: <https://w3id.org/legal-ontology-rag/ontology/normative-rule#> .
        @prefix lortc: <https://w3id.org/legal-ontology-rag/ontology/temporal-change#> .
        ex:rule a lornr:PenaltyRule ; lortc:validDuring ex:period .
        """,
        format="turtle",
    )

    rules = InstanceRetriever(graph).retrieve_batch([EX.rule])

    assert len(rules) == 1
    assert rules[0].valid_from is None
    assert rules[0].valid_to is None


def test_temporal_resolver_filters_rules(test_fixture_path: Path) -> None:
    """Open intervals are reported as indeterminate rather than ongoing."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)
    temporal_index = TemporalIndex(ontology_index.graph)

    rule_uris = list(temporal_index.all_rules())[:2]
    retriever = InstanceRetriever(ontology_index.graph)
    rules = retriever.retrieve_batch(rule_uris)

    resolver = TemporalResolver()
    resolution = resolver.partition(rules, date(2025, 6, 1))

    assert resolution.valid == ()
    assert resolution.invalid == ()
    assert resolution.indeterminate == tuple(rules)


def test_sparql_search_finds_instances(test_fixture_path: Path) -> None:
    """SPARQLSearch finds instances by keywords."""
    ontology_paths = [test_fixture_path / "normative_rule_valid.ttl"]
    ontology_index = OntologyIndex(ontology_paths)

    search = SPARQLSearch(ontology_index.graph)
    instances = search.search(["biển số"])

    assert len(instances) >= 0


def test_temporal_resolver_uses_half_open_boundaries(test_fixture_path: Path) -> None:
    """A replacement date excludes the old rule and includes the bounded new rule."""
    ontology_index = OntologyIndex([test_fixture_path / "normative_rule_valid.ttl"])
    rule_uri = next(iter(TemporalIndex(ontology_index.graph).all_rules()))
    base = InstanceRetriever(ontology_index.graph).retrieve_batch([rule_uri])[0]
    boundary = date(2026, 8, 15)
    old = replace(base, uri=URIRef("urn:old"), valid_from=date(2025, 1, 1), valid_to=boundary)
    new = replace(base, uri=URIRef("urn:new"), valid_from=boundary, valid_to=date(2027, 1, 1))
    unknown_end = replace(base, uri=URIRef("urn:unknown-end"), valid_from=date(2025, 1, 1), valid_to=None)
    unknown_start = replace(base, uri=URIRef("urn:unknown-start"), valid_from=None, valid_to=date(2027, 1, 1))

    resolver = TemporalResolver()
    resolution = resolver.partition([old, new, unknown_end, unknown_start], boundary)

    assert resolution.valid == (new,)
    assert resolution.invalid == (old,)
    assert resolution.indeterminate == (unknown_end, unknown_start)
    assert resolver.filter_valid([old, new, unknown_end, unknown_start], boundary) == [new]


def test_chunk_linker_uses_hierarchical_evidence_and_rdf_type() -> None:
    """A hierarchical result links leaf evidence only to RDF-typed rules."""
    graph = Graph()
    graph.add((EX.chunk, DCTERMS.identifier, Literal("leaf-1")))
    graph.add((EX.rule, RDF.type, LORNR.PenaltyRule))
    graph.add((EX.rule, LORNR.hasApplicability, EX.component))
    graph.add((EX.component, LORSP.sourcedFromChunk, EX.chunk))
    graph.add((URIRef("https://example.test/rule/not-a-rule"), LORSP.sourcedFromChunk, EX.chunk))
    chunk = RetrievedChunk(
        chunk_id="parent-unit",
        text="text",
        source_name="source",
        kind="section",
        score=1.0,
        evidence_chunk_ids=("leaf-1",),
    )

    linked = ChunkLinker(ChunkIndex(graph)).link([chunk])

    assert len(linked) == 1
    assert linked[0].uri == EX.rule
    assert linked[0].type == "PenaltyRule"
    assert linked[0].source_chunk_ids == ("leaf-1",)


def test_sparql_search_keeps_all_sources_before_result_limit() -> None:
    """Row multiplicity cannot truncate provenance before instance ranking."""
    graph = Graph()
    graph.add((EX.rule, RDF.type, LORNR.PenaltyRule))
    graph.add((EX.rule, RDFS.label, Literal("Vượt tốc độ nghiêm trọng", lang="vi")))
    expected_chunk_ids = tuple(f"chunk-{index:03d}" for index in range(120))
    for chunk_id in expected_chunk_ids:
        chunk_uri = URIRef(f"https://example.test/{chunk_id}")
        graph.add((EX.rule, LORSP.sourcedFromChunk, chunk_uri))
        graph.add((chunk_uri, DCTERMS.identifier, Literal(chunk_id)))

    instances = SPARQLSearch(graph).search(["vượt tốc độ nghiêm trọng"])

    assert len(instances) == 1
    assert instances[0].source_chunk_ids == expected_chunk_ids
