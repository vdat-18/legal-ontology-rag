"""Tests for one-shot adaptive RAG orchestration."""

from __future__ import annotations

import pytest
from loguru import logger

from legal_rag.adaptive_rag.models import AdaptiveQueryPlan, AdaptiveRoute
from legal_rag.adaptive_rag.pipeline import AdaptiveRAGPipeline
from legal_rag.naive_rag.models import RetrievedChunk, SearchMode
from legal_rag.query_rewriting import RewrittenQuery
from legal_rag.settings import AdaptiveRagSettings, MilvusSettings, NaiveRagSettings, Settings


class _FakeRetriever:
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, SearchMode]] = []
        self.close_calls = 0

    def retrieve(self, query: str, top_k: int, mode: SearchMode) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, mode))
        return self.results.get(query, [])[:top_k]

    def close(self) -> None:
        self.close_calls += 1


class _FakeGenerator:
    def __init__(self) -> None:
        self.query = ""
        self.chunks: list[object] = []

    def generate(self, query_vi: str, chunks: list[object]) -> str:
        self.query = query_vi
        self.chunks = list(chunks)
        return "answer"


class _FakePlanner:
    def __init__(self, plan: AdaptiveQueryPlan) -> None:
        self.result = plan
        self.queries: list[str] = []

    def plan(self, query: str) -> AdaptiveQueryPlan:
        self.queries.append(query)
        return self.result


class _FakeRewriter:
    def __init__(self, rewritten: str) -> None:
        self.rewritten = rewritten
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def rewrite(self, query: str, history: tuple[tuple[str, str], ...]) -> RewrittenQuery:
        self.calls.append((query, history))
        return RewrittenQuery(original=query, rewritten=self.rewritten)


def _pipeline(
    retriever: _FakeRetriever,
    planner: _FakePlanner,
    generator: _FakeGenerator,
    *,
    query_rewriter: _FakeRewriter | None = None,
    adaptive_settings: AdaptiveRagSettings | None = None,
) -> AdaptiveRAGPipeline:
    settings = Settings("http://localhost:11434/v1", "test-key", "test-model", 30)
    naive_settings = NaiveRagSettings(embedding_profile="bge_m3", top_k=5)
    return AdaptiveRAGPipeline(
        settings,
        MilvusSettings(uri="unused", database="unused"),
        naive_settings,
        adaptive_settings or AdaptiveRagSettings(),
        retriever=retriever,
        planner=planner,
        generation_client=generator,
        query_rewriter=query_rewriter or _FakeRewriter("unused"),
    )


def _chunk(chunk_id: str, text: str, score: float = 1.0, **overrides) -> RetrievedChunk:
    values = {
        "chunk_id": chunk_id,
        "text": text,
        "source_name": "document.docx",
        "kind": "paragraph_group",
        "score": score,
    }
    values.update(overrides)
    return RetrievedChunk(**values)


def test_direct_route_retrieves_once_and_keeps_base_scores():
    """Direct routing delegates once to the configured base retriever."""
    hit = _chunk("chunk-1", "context", 0.8)
    retriever = _FakeRetriever({"question": [hit]})
    planner = _FakePlanner(AdaptiveQueryPlan(AdaptiveRoute.DIRECT))
    generator = _FakeGenerator()
    pipeline = _pipeline(retriever, planner, generator)

    result = pipeline.run("question", top_k=3, rewrite_query=False)

    assert retriever.calls == [("question", 3, SearchMode.HYBRID_RERANK)]
    assert result.sources == (hit,)
    assert result.sources[0].score == 0.8
    assert result.route == AdaptiveRoute.DIRECT
    assert generator.query == "question"


def test_multi_hop_route_fuses_original_and_subquery_rankings():
    """Multi-hop retrieval uses bounded legs and shared deterministic RRF."""
    original_shared = _chunk("shared", "original representative")
    retriever = _FakeRetriever(
        {
            "question": [original_shared, _chunk("original-only", "original")],
            "subquery one": [_chunk("shared", "subquery representative"), _chunk("subquery-only", "subquery")],
            "subquery two": [],
        }
    )
    plan = AdaptiveQueryPlan(AdaptiveRoute.MULTI_HOP, ("subquery one", "subquery two"))
    generator = _FakeGenerator()
    pipeline = _pipeline(retriever, _FakePlanner(plan), generator)

    result = pipeline.run("question", top_k=3, rewrite_query=False)

    assert retriever.calls == [
        ("question", 10, SearchMode.HYBRID_RERANK),
        ("subquery one", 10, SearchMode.HYBRID_RERANK),
        ("subquery two", 10, SearchMode.HYBRID_RERANK),
    ]
    assert [source.chunk_id for source in result.sources] == ["shared", "original-only", "subquery-only"]
    assert [source.score for source in result.sources] == pytest.approx([2 / 61, 1 / 62, 1 / 62])
    assert result.sources[0].text == "original representative"
    assert generator.query == "question"


def test_multi_hop_preserves_hierarchical_unit_evidence():
    """Article units fuse by effective unit ID without losing source chunk provenance."""
    first = _chunk(
        "seed-1",
        "expanded article",
        unit_id="article:1",
        unit_kind="article",
        evidence_chunk_ids=("seed-1", "neighbor"),
    )
    duplicate = _chunk("seed-2", "same article", unit_id="article:1", unit_kind="article")
    retriever = _FakeRetriever({"question": [first], "one": [duplicate], "two": []})
    plan = AdaptiveQueryPlan(AdaptiveRoute.MULTI_HOP, ("one", "two"))
    pipeline = _pipeline(retriever, _FakePlanner(plan), _FakeGenerator())

    result = pipeline.run("question", top_k=2, rewrite_query=False)

    assert len(result.sources) == 1
    assert result.sources[0].effective_unit_id == "article:1"
    assert result.sources[0].effective_evidence_chunk_ids == ("seed-1", "neighbor")


def test_history_is_rewritten_before_planning_and_generation_keeps_raw_query():
    """Conversation context affects planning and retrieval but not the user-facing question."""
    retriever = _FakeRetriever({"standalone question": [_chunk("chunk-1", "context")]})
    planner = _FakePlanner(AdaptiveQueryPlan(AdaptiveRoute.DIRECT))
    generator = _FakeGenerator()
    rewriter = _FakeRewriter("standalone question")
    pipeline = _pipeline(retriever, planner, generator, query_rewriter=rewriter)
    history = (("previous", "answer"),)

    result = pipeline.run("follow-up", history=history)

    assert rewriter.calls == [("follow-up", history)]
    assert planner.queries == ["standalone question"]
    assert retriever.calls[0][0] == "standalone question"
    assert generator.query == "follow-up"
    assert result.rewritten_query == "standalone question"


def test_empty_multi_hop_results_generate_with_empty_context():
    """No-hit plans preserve the existing grounded generation boundary."""
    retriever = _FakeRetriever({})
    generator = _FakeGenerator()
    plan = AdaptiveQueryPlan(AdaptiveRoute.MULTI_HOP, ("one", "two"))
    pipeline = _pipeline(retriever, _FakePlanner(plan), generator)

    result = pipeline.run("question", rewrite_query=False)

    assert result.sources == ()
    assert generator.chunks == []


def test_pipeline_closes_retriever_once():
    """Adaptive pipeline owns one shared retrieval connection across all legs."""
    retriever = _FakeRetriever({})
    pipeline = _pipeline(retriever, _FakePlanner(AdaptiveQueryPlan(AdaptiveRoute.DIRECT)), _FakeGenerator())

    pipeline.close()

    assert retriever.close_calls == 1


def test_pipeline_does_not_log_queries():
    """Route diagnostics do not expose legal questions or subquery text."""
    query = "sensitive legal question"
    retriever = _FakeRetriever({})
    pipeline = _pipeline(retriever, _FakePlanner(AdaptiveQueryPlan(AdaptiveRoute.DIRECT)), _FakeGenerator())
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        pipeline.run(query, rewrite_query=False)
    finally:
        logger.remove(sink_id)

    assert all(query not in message for message in messages)
