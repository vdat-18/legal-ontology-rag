"""Tests for Corrective RAG configuration loading."""

from __future__ import annotations

import pytest

from legal_rag.corrective_rag.generate.config import RAGConfig


def test_rag_config_requires_llm_provider_settings() -> None:
    """CRAG config must not use hardcoded LLM provider defaults."""
    with pytest.raises(KeyError, match="LLM_BASE_URL"):
        RAGConfig.from_env({})


def test_rag_config_preserves_configured_llm_provider() -> None:
    """Configured OpenAI-compatible provider values pass through unchanged."""
    config = RAGConfig.from_env(
        {
            "LLM_BASE_URL": "https://provider.test/v1/",
            "LLM_API_KEY": "provider-key",
            "LLM_MODEL": "provider/model",
            "LLM_TIMEOUT_SECONDS": "12",
            "EMBEDDING_PROFILE": "bge_m3",
        }
    )

    assert config.llm_base_url == "https://provider.test/v1"
    assert config.llm_api_key == "provider-key"
    assert config.llm_model == "provider/model"
    assert config.llm_timeout_seconds == 12.0


def test_rag_config_invalid_numeric_value_raises() -> None:
    """Invalid numeric CRAG configuration must not silently fall back to defaults."""
    environment = {
        "LLM_BASE_URL": "https://provider.test/v1",
        "LLM_API_KEY": "provider-key",
        "LLM_MODEL": "provider/model",
        "LLM_TIMEOUT_SECONDS": "12",
        "EMBEDDING_PROFILE": "bge_m3",
        "CRAG_EVALUATOR_UPPER_THRESHOLD": "not-a-float",
    }

    with pytest.raises(ValueError, match="not-a-float"):
        RAGConfig.from_env(environment)
