"""Tests for Corrective RAG API diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from legal_rag.corrective_rag.api import server


def test_query_endpoint_passes_structured_history_to_pipeline() -> None:
    """API follow-ups preserve question-answer boundaries for query rewriting."""
    pipeline = _HistoryPipeline()
    app = server.create_app(pipeline_factory=lambda: pipeline, config_factory=_fake_config)

    with TestClient(app) as client:
        response = client.post(
            "/crag/query",
            json={
                "query": "Còn xe máy thì sao?",
                "limit": 3,
                "history": [{"question": "Ô tô vượt đèn đỏ bị phạt gì?", "answer": "Phạt tiền."}],
            },
        )

    assert response.status_code == 200
    assert pipeline.calls == [
        (
            "Còn xe máy thì sao?",
            3,
            (("Ô tô vượt đèn đỏ bị phạt gì?", "Phạt tiền."),),
        )
    ]
    assert response.json()["rewritten_query"] == "Xe máy vượt đèn đỏ bị phạt gì?"


def test_web_search_endpoint_reports_enabled_and_results(monkeypatch) -> None:
    """Direct web-search diagnostics do not require loading the CRAG pipeline."""
    monkeypatch.setattr(server, "WebSearchProvider", _FakeWebSearchProvider)
    app = server.create_app(
        pipeline_factory=_raise_if_pipeline_loads,
        config_factory=_fake_config,
    )

    with TestClient(app) as client:
        response = client.post("/crag/web-search", json={"query": "vuot den do", "limit": 1})
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "query": "vuot den do",
        "limit": 1,
        "enabled": True,
        "results": [{"source_name": "source", "url": "https://example.test", "text": "web text"}],
    }
    assert health.json()["pipeline_loaded"] is False
    assert health.json()["web_search_enabled"] is True


def _fake_config():
    """Return config fields used by API diagnostics."""
    return SimpleNamespace(
        embedding_profile="bge_m3",
        llm_model="provider/model",
        milvus_uri="http://localhost:19530",
        web_search_enabled=True,
    )


def _raise_if_pipeline_loads():
    """Fail when a diagnostic endpoint unexpectedly loads the pipeline."""
    msg = "pipeline should not load"
    raise AssertionError(msg)


class _FakeWebSearchProvider:
    """Web-search fake for API tests."""

    def __init__(self, config) -> None:
        self.config = config
        self.closed = False

    def search(self, query: str) -> list[dict[str, str]]:
        """Return stable web documents."""
        return [
            {"source_name": "source", "url": "https://example.test", "text": "web text"},
            {"source_name": "source 2", "url": "https://example.test/2", "text": "web text 2"},
        ]

    def close(self) -> None:
        """Close fake resources."""
        self.closed = True


class _HistoryPipeline:
    """Pipeline fake recording structured multi-turn API input."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[tuple[str, str], ...]]] = []

    def run(
        self,
        query: str,
        limit: int,
        history: tuple[tuple[str, str], ...],
    ) -> dict[str, object]:
        """Record the request and return a complete CRAG response."""
        self.calls.append((query, limit, history))
        return {
            "query": query,
            "rewritten_query": "Xe máy vượt đèn đỏ bị phạt gì?",
            "reformulated_query": None,
            "status": "SUCCESS_CORRECT",
            "crag_action": "CORRECT",
            "retrieved_count": 0,
            "second_retrieved_count": 0,
            "web_search_count": 0,
            "internal_knowledge_count": 0,
            "external_knowledge_count": 0,
            "relevant_count": 0,
            "graded_chunks": [],
            "answer": "answer",
        }

    def close(self) -> None:
        """Close fake resources."""
