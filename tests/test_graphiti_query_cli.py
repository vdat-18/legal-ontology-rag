"""Boundary-integration tests for the Graphiti query command."""

from __future__ import annotations

import asyncio

import pytest
from loguru import logger

from legal_rag import graphiti_query_cli
from legal_rag.graphiti_rag import pipeline as pipeline_module
from legal_rag.graphiti_rag.pipeline import GraphitiRAGResult
from legal_rag.settings import GraphitiSettings, Settings


def test_query_run_preserves_unknown_corpus_and_closes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI passes unknown coverage as None, emits the answer, and always closes the pipeline."""
    captured: dict[str, object] = {}
    sensitive_query = "sensitive-person-case-42"

    class FakePipeline:
        """Record CLI orchestration without constructing external clients."""

        def __init__(
            self,
            settings: object,
            graphiti_settings: object,
            group_id: str,
            known_so_hieu: frozenset[str] | None,
        ) -> None:
            captured.update(
                settings=settings,
                graphiti_settings=graphiti_settings,
                group_id=group_id,
                known_so_hieu=known_so_hieu,
            )

        async def run(self, query: str, max_facts: int | None = None) -> GraphitiRAGResult:
            """Return one deterministic CLI result."""
            captured.update(query=query, max_facts=max_facts)
            return GraphitiRAGResult(query, "formal legal query", "safe answer")

        async def close(self) -> None:
            """Record lifecycle cleanup."""
            captured["closed"] = True

    monkeypatch.setattr(Settings, "from_environment", classmethod(lambda cls: object()))
    monkeypatch.setattr(GraphitiSettings, "from_environment", classmethod(lambda cls: object()))
    monkeypatch.setattr(pipeline_module, "GraphitiRAGPipeline", FakePipeline)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        asyncio.run(graphiti_query_cli._run(sensitive_query, "group-1", None, 4))
    finally:
        logger.remove(sink_id)

    assert captured["known_so_hieu"] is None
    assert captured["group_id"] == "group-1"
    assert captured["max_facts"] == 4
    assert captured["closed"] is True
    assert any("safe answer" in message for message in messages)
    assert all(sensitive_query not in message for message in messages)
    assert all("formal legal query" not in message for message in messages)
