"""Boundary-integration tests for the Graphiti rewrite-search-generate pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from loguru import logger

from legal_rag.graphiti_rag import pipeline as pipeline_module
from legal_rag.graphiti_rag.pipeline import GraphitiRAGPipeline
from legal_rag.graphiti_rag.retriever import FactSource, RetrievedFact
from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings


class _FakeGraphiti:
    """Record whether the pipeline releases its graph connection."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        """Record connection closure without external I/O."""
        self.closed = True


class _FakeRewriteClient:
    """Return one fixed formal query and record the original input."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.histories: list[list[object]] = []

    async def rewrite(self, query: str, history: Sequence[object] = ()) -> str:
        """Return a deterministic rewritten query, recording the history it was given."""
        self.calls.append(query)
        self.histories.append(list(history))
        return "formal legal query"


class _FakeGenerationClient:
    """Return one fixed answer and record selected context."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[object]]] = []

    def generate(self, query_vi: str, chunks: Sequence[object]) -> str:
        """Return a deterministic answer containing one legal citation."""
        self.calls.append((query_vi, list(chunks)))
        return "Answer under 100/2019/ND-CP."


class _FakeCitationClient:
    """Return configured citations and record whether coverage checking ran."""

    def __init__(self, citations: list[str]) -> None:
        self.calls: list[str] = []
        self._citations = citations

    def extract_citations(self, text: str) -> list[str]:
        """Return configured document numbers."""
        self.calls.append(text)
        return list(self._citations)


def _settings() -> tuple[Settings, Neo4jSettings, GraphitiSettings]:
    """Build valid settings without loading process environment variables."""
    return (
        Settings(
            llm_base_url="http://localhost:20128/v1",
            llm_api_key="test-key",
            llm_model="provider/test-model",
            llm_timeout_seconds=30,
        ),
        Neo4jSettings(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="example-password",  # noqa: S106
        ),
        GraphitiSettings(
            embedding_profile="qwen3_0_6b",
            max_context_facts=1,
        ),
    )


def _build_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    known_document_numbers: frozenset[str] | None,
    citation_client: _FakeCitationClient,
) -> tuple[GraphitiRAGPipeline, _FakeGraphiti, _FakeRewriteClient, _FakeGenerationClient, list[tuple[object, ...]]]:
    """Replace external boundaries and construct one deterministic pipeline."""
    graphiti = _FakeGraphiti()
    rewrite_client = _FakeRewriteClient()
    generation_client = _FakeGenerationClient()
    search_calls: list[tuple[object, ...]] = []
    facts = [
        RetrievedFact("Fact under 100/2019/ND-CP.", (FactSource("chunk-1", "doc-1.docx"),)),
        RetrievedFact("Lower ranked fact.", (FactSource("chunk-2", "doc-2.docx"),)),
    ]

    async def fake_search(*args: object, **kwargs: object) -> list[RetrievedFact]:
        search_calls.append((*args, kwargs))
        return facts

    monkeypatch.setattr(pipeline_module, "load_configured_embedding_profile", lambda _profile: object())
    monkeypatch.setattr(pipeline_module, "build_graphiti", lambda *_args: graphiti)
    monkeypatch.setattr(pipeline_module, "BamlQueryRewriteClient", lambda *_args: rewrite_client)
    monkeypatch.setattr(pipeline_module, "BamlGenerationClient", lambda *_args: generation_client)
    monkeypatch.setattr(pipeline_module, "BamlDocumentCitationClient", lambda *_args: citation_client)
    monkeypatch.setattr(pipeline_module, "search_facts", fake_search)
    settings, neo4j_settings, graphiti_settings = _settings()
    pipeline = GraphitiRAGPipeline(
        settings,
        neo4j_settings,
        graphiti_settings,
        "group-1",
        known_document_numbers=known_document_numbers,
    )
    return pipeline, graphiti, rewrite_client, generation_client, search_calls


def test_pipeline_skips_citation_coverage_when_corpus_is_unknown_and_does_not_log_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing corpus metadata avoids a false caveat, an LLM call, and sensitive query logging."""
    citation_client = _FakeCitationClient(["100/2019/ND-CP"])
    sensitive_query = "sensitive-person-case-42"
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        pipeline, graphiti, rewrite_client, generation_client, search_calls = _build_pipeline(
            monkeypatch, None, citation_client
        )
        result = asyncio.run(pipeline.run(sensitive_query))
        asyncio.run(pipeline.close())
    finally:
        logger.remove(sink_id)

    assert rewrite_client.calls == [sensitive_query]
    assert search_calls[0][1:] == (
        "formal legal query",
        "group-1",
        30,
        {"expand_around_top_fact": True, "rerank_with_cross_encoder": False},
    )
    assert generation_client.calls[0][0] == sensitive_query
    assert len(generation_client.calls[0][1]) == 1
    assert result.unindexed_citations == ()
    assert citation_client.calls == []
    assert graphiti.closed is True
    assert all(sensitive_query not in message for message in messages)
    assert all("formal legal query" not in message for message in messages)


def test_pipeline_checks_citation_coverage_when_corpus_is_known(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly known corpus runs coverage checking and reports outside citations."""
    citation_client = _FakeCitationClient(["100/2019/ND-CP"])
    pipeline, _, _, _, _ = _build_pipeline(monkeypatch, frozenset(), citation_client)

    result = asyncio.run(pipeline.run("query"))

    assert result.unindexed_citations == ("100/2019/ND-CP",)
    assert len(citation_client.calls) == 1
