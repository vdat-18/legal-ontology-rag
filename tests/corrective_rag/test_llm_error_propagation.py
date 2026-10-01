"""Tests for CRAG LLM error propagation."""

from __future__ import annotations

import http.client
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from baml_py.errors import BamlError
from fastapi.testclient import TestClient

from legal_rag.corrective_rag.api import server
from legal_rag.corrective_rag.generate import grader as grader_module
from legal_rag.corrective_rag.generate.generator import ResponseGenerator
from legal_rag.corrective_rag.generate.grader import CorrectiveGrader
from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline


def test_grader_propagates_provider_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluator must not convert provider errors into score 0."""
    monkeypatch.setattr(grader_module.BamlCaller, "execute", _raise_provider_error)
    grader = CorrectiveGrader(_llm_config())

    with pytest.raises(http.client.HTTPException):
        grader.score_document("vuot den do", "muc phat")


def test_grader_rejects_unparseable_llm_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluator must not convert malformed LLM output into score 0."""
    monkeypatch.setattr(grader_module.BamlCaller, "execute", _malformed_grading_output)
    grader = CorrectiveGrader(_llm_config())

    with pytest.raises(BamlError):
        grader.score_document("vuot den do", "muc phat")


def test_grader_rejects_out_of_range_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluator must not clamp invalid typed scores into the accepted range."""
    monkeypatch.setattr(grader_module.BamlCaller, "execute", _out_of_range_grading_output)
    grader = CorrectiveGrader(_llm_config())

    with pytest.raises(ValueError, match="between 0 and 1"):
        grader.score_document("vuot den do", "muc phat")


def test_generator_propagates_provider_errors() -> None:
    """Answer generation must not return an error string as a successful answer."""
    generator = ResponseGenerator(_llm_config())
    generator.client = _FailingLlmClient()

    with pytest.raises(httpx.ConnectError):
        generator.generate_answer("vuot den do", [{"source_name": "source", "text": "muc phat"}])


def test_generator_rejects_empty_provider_choices() -> None:
    """Answer generation must not return a successful fallback for invalid provider output."""
    generator = ResponseGenerator(_llm_config())
    generator.client = _JsonLlmClient({"choices": []})

    with pytest.raises(ValueError, match="choices"):
        generator.generate_answer("vuot den do", [{"source_name": "source", "text": "muc phat"}])


def test_generator_rejects_output_truncated_at_token_limit() -> None:
    """A partial answer is not accepted when the provider exhausts the configured budget."""
    generator = ResponseGenerator(_llm_config())
    generator.client = _JsonLlmClient(
        {"choices": [{"finish_reason": "length", "message": {"content": "partial answer"}}]}
    )

    with pytest.raises(ValueError, match="output-token limit"):
        generator.generate_answer("vuot den do", [{"source_name": "source", "text": "muc phat"}])


def test_reformulation_propagates_provider_errors() -> None:
    """Query reformulation must not silently fall back to the original query."""
    generator = ResponseGenerator(_llm_config())
    generator.client = _FailingLlmClient()

    with pytest.raises(httpx.ConnectError):
        generator.reformulate_query("vuot den do")


def test_generator_rejects_oversized_provider_response() -> None:
    """Answer generation stops reading when the configured byte bound is exceeded."""
    generator = ResponseGenerator(_llm_config(max_response_bytes=8))
    generator.client = _JsonLlmClient({"choices": [{"message": {"content": "answer"}}]})

    with pytest.raises(ValueError, match="response exceeded 8 bytes"):
        generator.generate_answer("vuot den do", [{"source_name": "source", "text": "muc phat"}])


def test_pipeline_does_not_return_success_when_grader_fails() -> None:
    """A grading failure stops the pipeline before SUCCESS_* metadata is created."""
    pipeline = object.__new__(CorrectiveRAGPipeline)
    pipeline.db = _RetrievedChunkStore()
    pipeline.grader = _FailingGrader()

    with pytest.raises(httpx.ConnectError):
        pipeline.run("vuot den do")


def test_query_endpoint_returns_500_when_pipeline_provider_fails() -> None:
    """The API must not report HTTP 200 when the CRAG pipeline fails on LLM/provider work."""
    app = server.create_app(pipeline_factory=_failing_pipeline, config_factory=_api_config)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/crag/query", json={"query": "vuot den do", "limit": 1})

    assert response.status_code == 500
    assert "SUCCESS" not in response.text


def _llm_config(*, max_response_bytes: int = 1_048_576) -> SimpleNamespace:
    """Return only the settings needed by LLM clients."""
    return SimpleNamespace(
        llm_base_url="http://provider.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=1,
        llm_extra_request_body={},
        evaluator_max_output_tokens=1024,
        evaluator_temperature=0.0,
        answer_max_output_tokens=2048,
        answer_temperature=0.0,
        query_reformulation_max_output_tokens=512,
        query_reformulation_temperature=0.3,
        max_response_bytes=max_response_bytes,
    )


def _api_config() -> SimpleNamespace:
    """Return config fields used by API diagnostics."""
    return SimpleNamespace(
        embedding_profile="bge_m3",
        llm_model="provider/model",
        milvus_uri="http://localhost:19530",
        web_search_enabled=True,
    )


def _failing_pipeline() -> object:
    """Return a pipeline fake that fails like an unavailable provider."""
    return _FailingPipeline()


def _raise_provider_error(self: object, request: object) -> str:
    """Fail like an unavailable provider."""
    raise http.client.HTTPException("fail")


def _malformed_grading_output(self: object, request: object) -> str:
    """Return text that cannot satisfy the BAML relevance schema."""
    return "khong phai json"


def _out_of_range_grading_output(self: object, request: object) -> str:
    """Return typed JSON with an invalid score value."""
    return '{"score": 2.0, "rationale": "invalid"}'


class _FailingLlmClient:
    """HTTP client fake that raises provider errors."""

    def stream(self, method: str, path: str, *, json: dict[str, object]) -> object:
        """Fail before a provider response is available."""
        raise httpx.ConnectError("fail")


class _JsonLlmClient:
    """HTTP client fake that returns a configured JSON response."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    @contextmanager
    def stream(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object],
    ) -> Iterator[httpx.Response]:
        """Return the configured provider payload."""
        request = httpx.Request(method, f"http://provider.test{path}")
        yield httpx.Response(200, json=self.payload, request=request)


class _RetrievedChunkStore:
    """Vector store fake with one retrieved chunk."""

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        """Return one chunk so grading is attempted."""
        return [
            {
                "chunk_id": "chunk-a",
                "source_name": "source-a",
                "text": "muc phat vuot den do",
                "score": 0.9,
            }
        ]


class _FailingGrader:
    """Grader fake that raises like an LLM provider failure."""

    def score_document(self, query: str, document_text: str) -> object:
        """Fail during grading."""
        raise httpx.ConnectError("fail")


class _FailingPipeline:
    """Pipeline fake that raises from its query path."""

    def run(self, query: str, limit: int) -> dict[str, object]:
        """Fail during provider-backed pipeline work."""
        raise httpx.ConnectError("fail")

    def close(self) -> None:
        """Close fake resources."""
