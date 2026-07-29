"""Boundary-integration tests for the Graphiti indexing command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from legal_rag import graphiti_index_cli
from legal_rag.datastores import neo4j as neo4j_module
from legal_rag.embedding import profiles as profile_module
from legal_rag.graphiti_rag import client as client_module
from legal_rag.graphiti_rag import effective_date as effective_date_module
from legal_rag.graphiti_rag import embedder as embedder_module
from legal_rag.graphiti_rag import indexer as indexer_module
from legal_rag.graphiti_rag.indexer import CorpusMetadata, IndexingSummary
from legal_rag.naive_rag import indexer as naive_indexer_module
from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings


class _FakeGraphiti:
    """Record Graphiti index setup and connection cleanup."""

    def __init__(self, api_key: str, close_error: Exception | None = None) -> None:
        self.api_key = api_key
        self.built_indices = False
        self.closed = False
        self._close_error = close_error

    async def build_indices_and_constraints(self) -> None:
        """Record schema setup."""
        self.built_indices = True

    async def close(self) -> None:
        """Record client cleanup."""
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _FakeNeo4jDriver:
    """Record direct driver cleanup."""

    def __init__(self, close_error: Exception | None = None) -> None:
        self.closed = False
        self._close_error = close_error

    async def close(self) -> None:
        """Record driver cleanup."""
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    build_results: list[_FakeGraphiti | Exception],
    driver: _FakeNeo4jDriver,
    summary: IndexingSummary,
    *,
    indexing_error: Exception | None = None,
) -> list[tuple[object, ...]]:
    """Replace external boundaries for one deterministic multi-client CLI run."""
    graphiti_settings = GraphitiSettings(
        embedding_profile="qwen3_0_6b",
    )
    metadata = CorpusMetadata(group_id="group-1", documents={})
    concurrent_calls: list[tuple[object, ...]] = []
    pending_build_results = iter(build_results)

    def fake_build_graphiti(*_args: object, **_kwargs: object) -> _FakeGraphiti:
        result = next(pending_build_results)
        if isinstance(result, Exception):
            raise result
        return result

    async def fake_index_concurrent(*args: object) -> IndexingSummary:
        concurrent_calls.append(args)
        if indexing_error is not None:
            raise indexing_error
        return summary

    async def fail_single_index(*_args: object) -> IndexingSummary:
        msg = "single-client index path must not run"
        raise AssertionError(msg)

    monkeypatch.setattr(Settings, "from_environment", classmethod(lambda cls: object()))
    monkeypatch.setattr(Neo4jSettings, "from_environment", classmethod(lambda cls: object()))
    monkeypatch.setattr(GraphitiSettings, "from_environment", classmethod(lambda cls: graphiti_settings))
    monkeypatch.setattr(naive_indexer_module, "load_chunks", lambda _path: [object()])
    monkeypatch.setattr(indexer_module, "load_corpus_metadata", lambda _path: metadata)
    monkeypatch.setattr(profile_module, "load_configured_embedding_profile", lambda _profile: object())
    monkeypatch.setattr(
        embedder_module.QwenGraphitiEmbedder, "from_profile", classmethod(lambda cls, profile: object())
    )
    monkeypatch.setattr(effective_date_module, "BamlEffectiveDateClient", lambda *_args: object())
    monkeypatch.setattr(
        client_module, "load_llm_api_keys", lambda: [f"key-{index}" for index in range(len(build_results))]
    )
    monkeypatch.setattr(client_module, "build_graphiti", fake_build_graphiti)
    monkeypatch.setattr(neo4j_module, "build_neo4j_driver", lambda _settings: driver)
    monkeypatch.setattr(indexer_module, "index_chunks_concurrent", fake_index_concurrent)
    monkeypatch.setattr(indexer_module, "index_chunks", fail_single_index)
    return concurrent_calls


def test_index_run_uses_all_api_keys_and_closes_every_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI wires concurrent indexing and releases all resources after a successful run."""
    clients = [_FakeGraphiti("key-0"), _FakeGraphiti("key-1")]
    driver = _FakeNeo4jDriver()
    expected_summary = IndexingSummary(3, 1, 2, 4)
    concurrent_calls = _configure_cli(monkeypatch, list(clients), driver, expected_summary)

    summary = asyncio.run(graphiti_index_cli._run(Path("chunks"), Path("corpus.json")))

    assert summary == expected_summary
    assert [client.api_key for client in clients] == ["key-0", "key-1"]
    assert clients[0].built_indices is True
    assert all(client.closed for client in clients)
    assert driver.closed is True
    assert len(concurrent_calls) == 1
    assert concurrent_calls[0][2] == clients


def test_index_run_closes_partial_resources_when_client_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially constructed run closes its driver and every client built before the failure."""
    client = _FakeGraphiti("key-0")
    driver = _FakeNeo4jDriver()
    construction_error = RuntimeError("client construction failed")
    _configure_cli(monkeypatch, [client, construction_error], driver, IndexingSummary(0, 0, 0, 0))

    with pytest.raises(RuntimeError, match="client construction failed"):
        asyncio.run(graphiti_index_cli._run(Path("chunks"), Path("corpus.json")))

    assert client.closed is True
    assert driver.closed is True


def test_index_run_preserves_indexing_error_and_reports_every_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup attempts every close while the original indexing exception remains primary."""
    clients = [_FakeGraphiti("key-0", RuntimeError("client close failed")), _FakeGraphiti("key-1")]
    driver = _FakeNeo4jDriver(RuntimeError("driver close failed"))
    indexing_error = ValueError("indexing failed")
    _configure_cli(
        monkeypatch,
        list(clients),
        driver,
        IndexingSummary(0, 0, 0, 0),
        indexing_error=indexing_error,
    )

    with pytest.raises(ValueError, match="indexing failed") as error:
        asyncio.run(graphiti_index_cli._run(Path("chunks"), Path("corpus.json")))

    assert all(client.closed for client in clients)
    assert driver.closed is True
    assert error.value.__notes__ == [
        "Cleanup failed: RuntimeError('client close failed')",
        "Cleanup failed: RuntimeError('driver close failed')",
    ]


def test_index_run_raises_cleanup_failures_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful indexing run still fails if any resource cannot be closed."""
    clients = [_FakeGraphiti("key-0", RuntimeError("client close failed")), _FakeGraphiti("key-1")]
    driver = _FakeNeo4jDriver()
    _configure_cli(monkeypatch, list(clients), driver, IndexingSummary(0, 0, 0, 0))

    with pytest.raises(BaseExceptionGroup, match="Graphiti cleanup failed") as error:
        asyncio.run(graphiti_index_cli._run(Path("chunks"), Path("corpus.json")))

    assert all(client.closed for client in clients)
    assert driver.closed is True
    assert error.value.subgroup(RuntimeError) is not None
