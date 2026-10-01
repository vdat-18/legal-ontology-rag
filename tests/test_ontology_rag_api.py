"""Tests for Ontology RAG API."""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus", reason="pymilvus required for API tests")

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from rdflib import URIRef

from legal_rag.ontology_rag.models import (
    ComponentProvenance,
    ConditionExpression,
    LegalConsequence,
    LegalException,
    OntologyTerm,
    ProvisionCitation,
)


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
    consequences: tuple[object, ...] = ()
    condition_expressions: tuple[object, ...] = ()
    exceptions: tuple[object, ...] = ()
    rule_provenance: object | None = None
    conduct_provenance: tuple[object, ...] = ()
    applicability_provenance: tuple[object, ...] = ()


@dataclass(frozen=True)
class _MockContext:
    """Minimal retrieved context for API response tests."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float


@dataclass(frozen=True)
class _MockResult:
    """Mock result for testing."""

    query: str
    answer: str
    rules: tuple[_MockPenaltyRule, ...]
    sources: tuple[_MockContext, ...]
    answer_contexts: tuple[_MockContext, ...]
    temporal_mode: str
    query_dates: tuple[date, ...]
    temporal_status: dict[str, str]
    candidate_count: int
    invalid_rule_count: int
    indeterminate_rule_count: int


@pytest.fixture
def _fake_result() -> _MockResult:
    """Fake result for testing."""
    provenance = ComponentProvenance(
        provision_uris=(URIRef("https://example.test/provision-5"),),
        provision_paths=("Điều 5",),
        source_chunk_ids=("chunk-001",),
    )
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
        consequences=(
            LegalConsequence(
                uri=URIRef("https://example.test/consequence-1"),
                kind="MonetaryPenalty",
                label_vi="Phạt tiền",
                minimum_amount=Decimal("400000"),
                maximum_amount=Decimal("600000"),
                currency="VND",
                affected_targets=(OntologyTerm(URIRef("https://example.test/motorcycle"), "Xe máy"),),
                provenance=provenance,
            ),
        ),
        condition_expressions=(
            ConditionExpression(
                uri=URIRef("https://example.test/condition-1"),
                kind="Atomic",
                label_vi="Người điều khiển xe máy",
                provenance=provenance,
            ),
        ),
        exceptions=(
            LegalException(
                uri=URIRef("https://example.test/exception-1"),
                citations=(
                    ProvisionCitation(
                        uri=URIRef("https://example.test/citation-1"),
                        citation_text_vi="trừ trường hợp quy định tại Điều 6",
                        referenced_provision_paths=("Điều 6",),
                    ),
                ),
                excluded_provision_paths=("Điều 6",),
                provenance=provenance,
            ),
        ),
        rule_provenance=provenance,
    )
    return _MockResult(
        query="Không đội mũ bảo hiểm",
        answer="Phạt tiền từ 400000 đến 600000 VND, trừ 2 điểm.",
        rules=(rule,),
        sources=(_MockContext("chunk-001", "source text", "document.docx", "paragraph", 0.9),),
        answer_contexts=(_MockContext("rule:001", "rule context", "Điều 5", "ontology", 1.0),),
        temporal_mode="as_of",
        query_dates=(date(2025, 6, 1),),
        temporal_status={"rule:001": "valid"},
        candidate_count=3,
        invalid_rule_count=1,
        indeterminate_rule_count=0,
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
    assert rule["subject_kind"] == "NaturalPerson"
    assert rule["roles"] == ["Người lái xe"]
    assert rule["source_chunk_ids"] == ["chunk-001"]
    assert rule["rule_provenance"]["source_chunk_ids"] == ["chunk-001"]
    assert rule["consequences"][0]["affected_targets"][0]["label_vi"] == "Xe máy"
    assert rule["condition_expressions"][0]["label_vi"] == "Người điều khiển xe máy"
    assert rule["exceptions"][0]["citations"][0]["referenced_provision_paths"] == ["Điều 6"]
    assert body["sources"][0]["chunk_id"] == "chunk-001"
    assert body["answer_contexts"][0]["text"] == "rule context"
    assert body["pipeline_metadata"]["temporal_mode"] == "as_of"
    assert body["pipeline_metadata"]["query_dates"] == ["2025-06-01"]
    assert body["pipeline_metadata"]["candidate_count"] == 3


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
    from legal_rag import ontology_rag_api

    assert ontology_rag_api._state.pipeline.run.call_args.kwargs["query_date"] == date(2025, 6, 1)
    assert response.json()["pipeline_metadata"]["requested_query_date"] == "2025-06-01"


def test_request_model_accepts_typed_date() -> None:
    """Internal callers can use the same date type received by the pipeline."""
    from legal_rag.ontology_rag_api import OntologyQueryRequest

    request = OntologyQueryRequest(query="test", query_date=date(2025, 6, 1))

    assert request.query_date == date(2025, 6, 1)


def test_query_invalid_date_rejected(client: TestClient) -> None:
    """Invalid date format returns 422."""
    response = client.post(
        "/query",
        json={"query": "test", "query_date": "invalid"},
        headers=_AUTH_HEADER,
    )
    assert response.status_code == 422


@pytest.mark.parametrize("query_date", ["20250601", "2025-06-01T00:00:00", "2025-6-1"])
def test_query_noncanonical_date_rejected(client: TestClient, query_date: str) -> None:
    """Date-like values outside exact YYYY-MM-DD form are rejected."""
    response = client.post(
        "/query",
        json={"query": "test", "query_date": query_date},
        headers=_AUTH_HEADER,
    )

    assert response.status_code == 422


def test_empty_query_rejected(client: TestClient) -> None:
    """Empty query string is rejected by validation."""
    response = client.post("/query", json={"query": ""}, headers=_AUTH_HEADER)
    assert response.status_code == 422


def test_main_starts_configured_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed API command delegates host, port, and reload settings to Uvicorn."""
    from legal_rag import ontology_rag_api

    run = MagicMock()
    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["legal-ontology-rag-api", "--host", "127.0.0.2", "--port", "8110", "--reload"],
    )

    ontology_rag_api.main()

    run.assert_called_once_with(
        "legal_rag.ontology_rag_api:app",
        host="127.0.0.2",
        port=8110,
        reload=True,
    )
