"""Tests for Corrective RAG API diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from legal_rag.corrective_rag.api import server


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
