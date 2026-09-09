"""Tests for Ontology RAG API."""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus", reason="pymilvus required for API tests")

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@dataclass(frozen=True)
class _MockPenaltyRule:
    """Mock penalty rule for testing."""

    uri: str
    conduct_label_vi: str
    vehicle_categories: tuple[str, ...]
    subject_kind: str | None
    roles: tuple[str, ...]
    monetary_penalty_min: Decimal | None
    monetary_penalty_max: Decimal | None
    currency: str | None
    point_value: int | None
    valid_from: date
    valid_to: date | None
    source_chunk_ids: tuple[str, ...]
    provision_path: str
    document_number: str


@dataclass(frozen=True)
class _MockResult:
    """Mock result for testing."""

    query: str
    answer: str
    rules: tuple[_MockPenaltyRule, ...]


@pytest.fixture
def _fake_result() -> _MockResult:
    """Fake result for testing."""
    rule = _MockPenaltyRule(
        uri="rule:001",
        conduct_label_vi="Không đội mũ bảo hiểm",
        vehicle_categories=("Xe máy",),
        subject_kind="NaturalPerson",
        roles=("Người lái xe",),
        monetary_penalty_min=Decimal("400000"),
        monetary_penalty_max=Decimal("600000"),
        currency="VND",
        point_value=2,
        valid_from=date(2025, 1, 1),
        valid_to=None,
        source_chunk_ids=("chunk-001",),
        provision_path="Điều 5",
        document_number="NĐ 17/2020",
    )
    return _MockResult(
        query="Không đội mũ bảo hiểm",
        answer="Phạt tiền từ 400000 đến 600000 VND, trừ 2 điểm.",
        rules=(rule,),
    )


_REQUIRED_ENV = {
    "LLM_BASE_URL": "http://localhost/v1",
    "LLM_API_KEY": "k",
    "LLM_MODEL": "m",
    "LLM_TIMEOUT_SECONDS": "10",
    "EMBEDDING_PROFILE": "bge_m3",
    "MILVUS_URI": "http://localhost:19530",
    "MILVUS_DATABASE": "legal_rag",
    "ONTOLOGY_PATHS": "ontology/test.ttl",
    "RAG_API_KEY": "test-api-key",
}


_AUTH_HEADER = {"X-Api-Key": "test-api-key"}


@pytest.fixture
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def client(_env: None, _fake_result: _MockResult) -> Iterator[TestClient]:
    """Test client with mocked pipeline."""
    from legal_rag import ontology_rag_api

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _fake_result

    with patch.object(ontology_rag_api, "_state") as mock_state:
        mock_state.pipeline = mock_pipeline
        mock_state.api_key = "test-api-key"
        mock_state.ontology_paths = [MagicMock()]

        client = TestClient(ontology_rag_api.app)
        yield client


def test_health(client: TestClient) -> None:
    """Health endpoint returns ok without auth."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_returns_answer_and_rules(client: TestClient) -> None:
    """Query returns answer and structured penalty rules."""
    response = client.post("/query", json={"query": "test"}, headers=_AUTH_HEADER)
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "test"
    assert body["answer"]
    assert len(body["rules"]) == 1
    rule = body["rules"][0]
    assert rule["conduct"] == "Không đội mũ bảo hiểm"
    assert rule["monetary_penalty"]["min"] == "400000"
    assert rule["point_deduction"] == 2


def test_query_without_api_key_rejected(client: TestClient) -> None:
    """Query without API key returns 401 or 422."""
    response = client.post("/query", json={"query": "test"})
    assert response.status_code in (401, 422)


def test_query_with_custom_date(client: TestClient) -> None:
    """Query accepts query_date parameter."""
    response = client.post(
        "/query",
        json={"query": "test", "query_date": "2025-06-01"},
        headers=_AUTH_HEADER,
    )
    assert response.status_code == 200


def test_query_invalid_date_rejected(client: TestClient) -> None:
    """Invalid date format returns 422."""
    response = client.post(
        "/query",
        json={"query": "test", "query_date": "invalid"},
        headers=_AUTH_HEADER,
    )
    assert response.status_code == 422


def test_empty_query_rejected(client: TestClient) -> None:
    """Empty query string is rejected by validation."""
    response = client.post("/query", json={"query": ""}, headers=_AUTH_HEADER)
    assert response.status_code == 422
