"""Integration tests for OntologyRAGPipeline."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdflib import URIRef

from legal_rag.datastores.milvus import MilvusCorpusFingerprint
from legal_rag.naive_rag.models import RetrievedChunk, SearchMode
from legal_rag.ontology_rag.context.context_builder import ContextBuilder
from legal_rag.ontology_rag.models import OntologyInstance, PenaltyRule
from legal_rag.ontology_rag.pipeline import OntologyRAGPipeline
from legal_rag.ontology_rag.query_planner import OntologyQueryPlan, TemporalQueryMode
from legal_rag.ontology_rag.retrieval.temporal_resolver import TemporalResolver
from legal_rag.settings import NaiveRagSettings


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
    resolution = resolver.partition(rules, date(2025, 6, 1))

    assert len(rules) > 0
    assert len(resolution.valid) + len(resolution.indeterminate) > 0


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


def _rule(rule_id: str, valid_from: date | None, valid_to: date | None, chunk_id: str) -> PenaltyRule:
    """Build one temporally distinct rule for orchestration tests."""
    return PenaltyRule(
        uri=URIRef(f"https://example.test/rule/{rule_id}"),
        conduct_label_vi=f"Hành vi {rule_id}",
        vehicle_categories=("Xe mô tô",),
        subject_kind="NaturalPerson",
        roles=("Người điều khiển",),
        monetary_penalty_min=Decimal("400000"),
        monetary_penalty_max=Decimal("600000"),
        currency="VND",
        point_value=None,
        valid_from=valid_from,
        valid_to=valid_to,
        source_chunk_ids=(chunk_id,),
        provision_path="Điều 7",
        document_number="168/2024/NĐ-CP",
    )


class _Planner:
    def __init__(self) -> None:
        self.reference_date = None

    def plan(self, query, history, reference_date):
        self.reference_date = reference_date
        return OntologyQueryPlan(
            search_query="câu hỏi pháp lý độc lập",
            temporal_mode=TemporalQueryMode.CURRENT,
        )


class _Retriever:
    def __init__(self) -> None:
        self.source_chunks = {
            f"chunk-{name}": RetrievedChunk(
                f"chunk-{name}",
                f"Nguyên văn {name}",
                "document.docx",
                "paragraph_group",
                1.0,
            )
            for name in ("old", "new", "unknown", "background")
        }

    def retrieve(self, query, top_k, mode):
        if mode == SearchMode.SEMANTIC:
            return [
                self.source_chunks["chunk-old"],
                self.source_chunks["chunk-background"],
                self.source_chunks["chunk-new"],
            ]
        return [self.source_chunks["chunk-unknown"]]

    def fetch_by_ids(self, chunk_ids):
        return [self.source_chunks[chunk_id] for chunk_id in chunk_ids]


class _Linker:
    def link(self, chunks):
        return [
            OntologyInstance(
                URIRef(f"https://example.test/rule/{chunk.chunk_id[6:]}"),
                "PenaltyRule",
                (chunk.chunk_id,),
            )
            for chunk in chunks
            if chunk.chunk_id in {"chunk-new", "chunk-unknown"}
        ]


class _SPARQLSearch:
    def search(self, keywords):
        return [OntologyInstance(URIRef("https://example.test/rule/old"), "PenaltyRule", ("chunk-old",))]


class _Fusion:
    def fuse(self, semantic_chunks, keyword_chunks, ontology_instances, **kwargs):
        return [
            ontology_instances[0],
            *kwargs["semantic_instances"],
            *kwargs["keyword_instances"],
        ]


class _InstanceRetriever:
    def __init__(self, rules):
        self.rules = {rule.uri: rule for rule in rules}

    def retrieve_batch(self, uris):
        return [self.rules[uri] for uri in uris]


class _GenerationClient:
    def __init__(self) -> None:
        self.contexts = ()
        self.query = None

    def generate(self, query_vi, chunks):
        self.query = query_vi
        self.contexts = tuple(chunks)
        return "Câu trả lời có căn cứ"


def test_pipeline_filters_before_top_k_and_generates_from_exact_sources() -> None:
    """Expired candidates reach neither the result budget nor generation fallback."""
    rules = (
        _rule("old", date(2025, 1, 1), date(2026, 8, 15), "chunk-old"),
        _rule("new", date(2026, 8, 15), date(2027, 1, 1), "chunk-new"),
        _rule("unknown", date(2025, 1, 1), None, "chunk-unknown"),
    )
    pipeline = object.__new__(OntologyRAGPipeline)
    pipeline._naive_rag_settings = NaiveRagSettings(embedding_profile="bge_m3", top_k=2)
    planner = _Planner()
    pipeline._query_planner = planner
    pipeline._retriever = _Retriever()
    pipeline._linker = _Linker()
    pipeline._sparql_search = _SPARQLSearch()
    pipeline._rrf_fusion = _Fusion()
    pipeline._instance_retriever = _InstanceRetriever(rules)
    pipeline._temporal_resolver = TemporalResolver()
    pipeline._context_builder = ContextBuilder()
    generation_client = _GenerationClient()
    pipeline._generation_client = generation_client

    result = pipeline.run(
        "câu hỏi",
        query_date=date(2026, 8, 15),
        top_k=2,
        reference_date=date(2026, 8, 14),
    )

    assert [rule.uri.rsplit("/", 1)[-1] for rule in result.rules] == ["new", "unknown"]
    assert result.invalid_rule_count == 1
    assert result.indeterminate_rule_count == 1
    assert result.candidate_count == 3
    assert result.query_dates == (date(2026, 8, 15),)
    assert planner.reference_date == date(2026, 8, 14)
    assert result.temporal_status[str(rules[1].uri)] == "valid"
    assert result.temporal_status[str(rules[2].uri)] == "indeterminate"
    assert generation_client.query == "câu hỏi pháp lý độc lập"
    assert "Nguyên văn new" in generation_client.contexts[0].text
    assert "Nguyên văn unknown" in generation_client.contexts[1].text
    assert all("Nguyên văn old" not in context.text for context in generation_client.contexts)
    assert any("Nguyên văn background" in context.text for context in generation_client.contexts)
    assert "chunk-old" not in {chunk.chunk_id for chunk in result.sources}
    assert "chunk-background" in {chunk.chunk_id for chunk in result.sources}

    comparison_dates = (date(2025, 1, 1), date(2026, 8, 15))
    comparison = pipeline.run(
        "câu hỏi",
        query_dates=comparison_dates,
        top_k=2,
        reference_date=date(2026, 8, 14),
    )

    assert [rule.uri.rsplit("/", 1)[-1] for rule in comparison.rules] == ["old", "new"]
    assert comparison.query_dates == comparison_dates
    assert comparison.temporal_mode == TemporalQueryMode.COMPARE_DATES.value

    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline.run(
            "câu hỏi",
            query_date=date(2025, 1, 1),
            query_dates=(date(2026, 8, 15),),
        )


def test_temporal_comparison_reserves_results_for_each_requested_date() -> None:
    """A highly ranked old rule cannot consume the whole budget of a date comparison."""
    old_first = _rule("old-first", date(2025, 1, 1), date(2026, 8, 15), "chunk-old")
    old_second = _rule("old-second", date(2025, 1, 1), date(2026, 8, 15), "chunk-old")
    new = _rule("new", date(2026, 8, 15), None, "chunk-new")
    pipeline = object.__new__(OntologyRAGPipeline)
    pipeline._temporal_resolver = TemporalResolver()

    selected, statuses, invalid_count, indeterminate_count = pipeline._select_temporal_rules(
        [old_first, old_second, new],
        (date(2026, 8, 14), date(2026, 8, 15)),
        2,
    )

    assert [rule.uri.rsplit("/", 1)[-1] for rule in selected] == ["old-first", "new"]
    assert statuses[str(old_first.uri)] == "2026-08-14=valid;2026-08-15=invalid"
    assert statuses[str(new.uri)] == "2026-08-14=invalid;2026-08-15=indeterminate"
    assert invalid_count == 0
    assert indeterminate_count == 1


def test_pipeline_fails_when_selected_rule_provenance_is_not_fetched() -> None:
    """A selected rule cannot expose evidence absent from the active corpus."""
    present = RetrievedChunk("present", "text", "source.docx", "paragraph_group", 1.0)
    pipeline = object.__new__(OntologyRAGPipeline)
    pipeline._retriever = SimpleNamespace(fetch_by_ids=lambda chunk_ids: [present])

    with pytest.raises(ValueError, match=r"Selected ontology rules.*count=1.*missing=\['missing'\]"):
        pipeline._fetch_source_chunks(("present", "missing"))


def test_preflight_fingerprints_verified_ontology_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ontology provenance IDs become part of the active backend fingerprint."""
    milvus_fingerprint = MilvusCorpusFingerprint(
        fingerprint_version="1",
        collection_name="ontology",
        row_count=2,
        corpus_sha256="corpus-digest",
        embedding_profile_id="profile",
        embedding_model_id="model",
        embedding_model_revision="revision",
    )
    pipeline = object.__new__(OntologyRAGPipeline)
    pipeline._retriever = SimpleNamespace(
        settings=SimpleNamespace(chunks_dir=Path("configured/chunks")),
        preflight_corpus=lambda: milvus_fingerprint,
    )
    pipeline._chunk_index = SimpleNamespace(all_chunk_ids=lambda: frozenset({"chunk-2", "chunk-1"}))
    monkeypatch.setattr(
        "legal_rag.ontology_rag.pipeline.load_chunks",
        lambda chunks_dir: (SimpleNamespace(chunk_id="chunk-1"), SimpleNamespace(chunk_id="chunk-2")),
    )

    fingerprint = pipeline.preflight_corpus().to_dict()

    assert fingerprint["corpus_sha256"] == "corpus-digest"
    assert fingerprint["ontology_provenance_fingerprint_version"] == "1"
    assert fingerprint["ontology_provenance_chunk_count"] == 2
    assert len(fingerprint["ontology_provenance_chunk_ids_sha256"]) == 64


def test_preflight_rejects_stale_ontology_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ontology files must not reference chunks outside the configured local corpus."""
    pipeline = object.__new__(OntologyRAGPipeline)
    pipeline._retriever = SimpleNamespace(
        settings=SimpleNamespace(chunks_dir=Path("configured/chunks")),
        preflight_corpus=lambda: SimpleNamespace(),
    )
    pipeline._chunk_index = SimpleNamespace(all_chunk_ids=lambda: frozenset({"present", "stale"}))
    monkeypatch.setattr(
        "legal_rag.ontology_rag.pipeline.load_chunks",
        lambda chunks_dir: (SimpleNamespace(chunk_id="present"),),
    )

    with pytest.raises(ValueError, match=r"Ontology provenance.*count=1.*missing=\['stale'\]"):
        pipeline.preflight_corpus()
