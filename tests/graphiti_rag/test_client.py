"""Tests for collecting configured LLM API keys for concurrent Graphiti indexing."""

from __future__ import annotations

import pytest

from legal_rag.graphiti_rag import client as client_module
from legal_rag.graphiti_rag.client import build_graphiti, load_llm_api_keys
from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings


def test_build_graphiti_uses_the_generic_openai_compatible_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """URL, key, model, and timeout are sufficient to construct the provider-neutral client."""
    captured: dict[str, object] = {}

    def fake_openai_client(**kwargs: object) -> tuple[str, dict[str, object]]:
        captured["openai"] = kwargs
        return "openai", kwargs

    def fake_llm_client(**kwargs: object) -> tuple[str, dict[str, object]]:
        captured["llm"] = kwargs
        return "llm", kwargs

    def fake_reranker(*args: object) -> tuple[str, tuple[object, ...]]:
        # The local reranker loads a model from disk, so the test must not construct a real one.
        return "reranker", args

    def fake_graphiti(**kwargs: object) -> dict[str, object]:
        return kwargs

    monkeypatch.setattr(client_module, "AsyncOpenAI", fake_openai_client)
    monkeypatch.setattr(client_module, "OpenAIGenericClient", fake_llm_client)
    monkeypatch.setattr(client_module, "LocalCrossEncoderReranker", fake_reranker)
    monkeypatch.setattr(client_module, "Graphiti", fake_graphiti)
    settings = Settings("https://example.test/v1", "test-key", "gpt-5.4", 60)
    neo4j_settings = Neo4jSettings("bolt://localhost:7687", "neo4j", "example-password")
    graphiti_settings = GraphitiSettings("qwen3_0_6b")
    embedder = object()

    graphiti = build_graphiti(  # type: ignore[arg-type]
        settings,
        neo4j_settings,
        graphiti_settings,
        object(),
        embedder=embedder,
    )

    assert captured["openai"] == {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "timeout": 60,
    }
    llm_arguments = captured["llm"]
    assert isinstance(llm_arguments, dict)
    assert llm_arguments["config"].model == "gpt-5.4"
    assert llm_arguments["structured_output_mode"] == "json_object"
    assert graphiti["embedder"] is embedder
    assert graphiti["uri"] == neo4j_settings.uri
    assert graphiti["user"] == neo4j_settings.user


def test_load_llm_api_keys_returns_only_the_primary_key_when_no_extras_are_set() -> None:
    """With no LLM_API_KEY_2.. configured, only the primary key is returned."""
    keys = load_llm_api_keys({"LLM_API_KEY": "sk-primary"})

    assert keys == ["sk-primary"]


def test_load_llm_api_keys_collects_contiguous_numbered_extras() -> None:
    """Extra keys LLM_API_KEY_2, _3, ... are collected in order until one is missing."""
    keys = load_llm_api_keys(
        {
            "LLM_API_KEY": "sk-primary",
            "LLM_API_KEY_2": "sk-two",
            "LLM_API_KEY_3": "sk-three",
        }
    )

    assert keys == ["sk-primary", "sk-two", "sk-three"]


def test_load_llm_api_keys_stops_at_the_first_gap() -> None:
    """A gap in the numbering (e.g. _2 missing but _3 present) stops the scan at the gap."""
    keys = load_llm_api_keys({"LLM_API_KEY": "sk-primary", "LLM_API_KEY_3": "sk-three"})

    assert keys == ["sk-primary"]
