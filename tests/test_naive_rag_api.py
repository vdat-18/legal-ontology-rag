"""Tests for the RAG API server."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@dataclass(frozen=True)
class _FakeChunk:
    chunk_id: str = "chunk-1"
    text: str = "Điều 5."
    source_name: str = "168.docx"
    kind: str = "paragraph_group"
    score: float = 0.85
    effective_unit_id: str = "chunk-1"
    unit_kind: str = "chunk"
    effective_evidence_chunk_ids: tuple[str, ...] = ("chunk-1",)


@dataclass(frozen=True)
class _FakeResult:
    query: str = "test"
    answer: str = "Phạt 4 triệu"
    sources: tuple = (_FakeChunk(),)


_REQUIRED_ENV = {
    "LLM_BASE_URL": "http://localhost/v1",
    "LLM_API_KEY": "k",
    "LLM_MODEL": "m",
    "LLM_TIMEOUT_SECONDS": "10",
    "EMBEDDING_PROFILE": "qwen3_0_6b",
    "MILVUS_URI": "http://localhost:19530",
    "MILVUS_DATABASE": "legal_rag",
    "NAIVE_RAG_MILVUS_COLLECTION_NAME": "naive_rag_chunks",
    "REDIS_URL": "redis://localhost:6379/0",
    "RAG_API_KEY": "test-api-key",
}

_AUTH_HEADER = {"X-Api-Key": "test-api-key"}


@pytest.fixture
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def client(_env: None) -> Iterator[TestClient]:
    """Test client with mocked pipeline and Redis."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _FakeResult()

    mock_redis = MagicMock()
    mock_redis.lrange.return_value = []

    with (
        patch("legal_rag.naive_rag.pipeline.NaiveRAGPipeline", return_value=mock_pipeline),
        patch("legal_rag.naive_rag_api.redis") as mock_redis_mod,
    ):
        mock_redis_mod.from_url.return_value = mock_redis
        from legal_rag.naive_rag_api import app

        with TestClient(app) as tc:
            yield tc


def test_health(client: TestClient) -> None:
    """Health endpoint returns ok without auth."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_returns_answer_sources_latency(client: TestClient) -> None:
    """Query returns answer, sources, and latency breakdown."""
    response = client.post("/query", json={"query": "Vượt đèn đỏ?"}, headers=_AUTH_HEADER)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Phạt 4 triệu"
    assert len(body["sources"]) == 1
    assert "pipeline_ms" in body["latency_ms"]
    assert "history_load_ms" in body["latency_ms"]


@pytest.mark.usefixtures("_env")
def test_query_passes_structured_history_to_the_pipeline() -> None:
    """The API keeps the current question separate from earlier conversation turns."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _FakeResult()
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = [b'{"query":"previous","answer":"previous answer"}']

    with (
        patch("legal_rag.naive_rag.pipeline.NaiveRAGPipeline", return_value=mock_pipeline),
        patch("legal_rag.naive_rag_api.redis") as mock_redis_mod,
    ):
        mock_redis_mod.from_url.return_value = mock_redis
        from legal_rag.naive_rag_api import app

        with TestClient(app) as test_client:
            response = test_client.post(
                "/query",
                json={"query": "follow-up"},
                headers={**_AUTH_HEADER, "X-Session-Id": "s-history"},
            )

    assert response.status_code == 200
    mock_pipeline.run.assert_called_once_with(
        "follow-up",
        top_k=None,
        search_mode=None,
        history=(("previous", "previous answer"),),
        rewrite_query=True,
    )


def test_query_without_api_key_rejected(client: TestClient) -> None:
    """Query without API key returns 401."""
    response = client.post("/query", json={"query": "test"})
    assert response.status_code in (401, 422)


def test_query_preserves_session_id(client: TestClient) -> None:
    """Session ID from header is returned in response."""
    response = client.post("/query", json={"query": "test"}, headers={**_AUTH_HEADER, "X-Session-Id": "s-123"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "s-123"


def test_query_generates_session_id(client: TestClient) -> None:
    """A session ID is generated when none is provided."""
    response = client.post("/query", json={"query": "test"}, headers=_AUTH_HEADER)
    assert response.status_code == 200
    assert len(response.json()["session_id"]) > 0


def test_session_isolation(client: TestClient) -> None:
    """Different sessions produce independent history."""
    client.post("/query", json={"query": "a"}, headers={**_AUTH_HEADER, "X-Session-Id": "s-a"})
    client.post("/query", json={"query": "b"}, headers={**_AUTH_HEADER, "X-Session-Id": "s-b"})
    resp_a = client.get("/history/s-a", headers=_AUTH_HEADER)
    resp_b = client.get("/history/s-b", headers=_AUTH_HEADER)
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200


def test_clear_history(client: TestClient) -> None:
    """Delete endpoint returns cleared status."""
    response = client.delete("/history/s-x", headers=_AUTH_HEADER)
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"


def test_empty_query_rejected(client: TestClient) -> None:
    """Empty query string is rejected by validation."""
    response = client.post("/query", json={"query": ""}, headers=_AUTH_HEADER)
    assert response.status_code == 422


def test_query_accepts_hybrid_rerank_mode(client: TestClient) -> None:
    """The API exposes the reranked Naive RAG variant."""
    response = client.post(
        "/query",
        json={"query": "Vượt đèn đỏ?", "search_mode": "hybrid_rerank"},
        headers=_AUTH_HEADER,
    )

    assert response.status_code == 200


def test_query_accepts_hierarchical_mode_and_returns_provenance(client: TestClient) -> None:
    """The API exposes hierarchical mode and source evidence identifiers."""
    response = client.post(
        "/query",
        json={"query": "Vượt đèn đỏ?", "search_mode": "hierarchical"},
        headers=_AUTH_HEADER,
    )

    assert response.status_code == 200
    [source] = response.json()["sources"]
    assert source["unit_id"] == "chunk-1"
    assert source["unit_kind"] == "chunk"
    assert source["evidence_chunk_ids"] == ["chunk-1"]


@pytest.mark.usefixtures("_env")
def test_redis_failure_crashes_startup() -> None:
    """Startup fails when Redis is unreachable."""
    mock_pipeline = MagicMock()
    mock_redis = MagicMock()
    mock_redis.ping.side_effect = Exception("connection refused")

    with (
        patch("legal_rag.naive_rag.pipeline.NaiveRAGPipeline", return_value=mock_pipeline),
        patch("legal_rag.naive_rag_api.redis") as mock_redis_mod,
    ):
        mock_redis_mod.from_url.return_value = mock_redis
        from legal_rag.naive_rag_api import app

        with pytest.raises(Exception, match="connection refused"), TestClient(app):
            pass


@pytest.mark.usefixtures("_env")
def test_store_turn_writes_to_redis() -> None:
    """Query stores the turn as JSON in Redis under the session key."""
    import json

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _FakeResult()
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = []

    with (
        patch("legal_rag.naive_rag.pipeline.NaiveRAGPipeline", return_value=mock_pipeline),
        patch("legal_rag.naive_rag_api.redis") as mock_redis_mod,
    ):
        mock_redis_mod.from_url.return_value = mock_redis
        from legal_rag.naive_rag_api import app

        with TestClient(app) as tc:
            tc.post("/query", json={"query": "xin chào"}, headers={**_AUTH_HEADER, "X-Session-Id": "s-write"})

    rpush_calls = [c for c in mock_redis.rpush.call_args_list if "rag:history:s-write" in str(c)]
    assert len(rpush_calls) == 1
    stored = json.loads(rpush_calls[0][0][1])
    assert stored["query"] == "xin chào"
    assert stored["answer"] == "Phạt 4 triệu"


@pytest.mark.usefixtures("_env")
def test_session_isolation_redis_keys() -> None:
    """Different sessions write to different Redis keys."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _FakeResult()
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = []

    with (
        patch("legal_rag.naive_rag.pipeline.NaiveRAGPipeline", return_value=mock_pipeline),
        patch("legal_rag.naive_rag_api.redis") as mock_redis_mod,
    ):
        mock_redis_mod.from_url.return_value = mock_redis
        from legal_rag.naive_rag_api import app

        with TestClient(app) as tc:
            tc.post("/query", json={"query": "a"}, headers={**_AUTH_HEADER, "X-Session-Id": "iso-a"})
            tc.post("/query", json={"query": "b"}, headers={**_AUTH_HEADER, "X-Session-Id": "iso-b"})

    rpush_keys = [c[0][0] for c in mock_redis.rpush.call_args_list]
    assert "rag:history:iso-a" in rpush_keys
    assert "rag:history:iso-b" in rpush_keys
