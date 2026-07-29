"""Tests for shared application settings compatibility."""

from __future__ import annotations

from legal_rag.settings import Settings


def test_shared_settings_load_from_openai_compatible_environment() -> None:
    """An OpenAI-compatible endpoint needs only URL, key, model, and timeout settings."""
    settings = Settings.from_environment(
        {
            "LLM_BASE_URL": "http://localhost:20128/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "provider/test-model",
            "LLM_TIMEOUT_SECONDS": "30",
        }
    )

    assert settings.llm_base_url == "http://localhost:20128/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_model == "provider/test-model"
    assert settings.llm_timeout_seconds == 30
