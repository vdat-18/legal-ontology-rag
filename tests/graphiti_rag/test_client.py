"""Tests for collecting configured LLM API keys for concurrent Graphiti indexing."""

from __future__ import annotations

import pytest
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from legal_rag.graphiti_rag.client import _LLM_CLIENT_CLASSES, DeepSeekNonThinkingClient, load_llm_api_keys
from legal_rag.settings import LlmProvider


def test_every_supported_provider_maps_to_an_llm_client_class() -> None:
    """No provider can be configured without a client class, so lookup never fails at runtime."""
    assert set(_LLM_CLIENT_CLASSES) == set(LlmProvider)


def test_only_deepseek_gets_the_thinking_mode_disabling_client() -> None:
    """DeepSeek's provider-specific extra_body field is never sent to another provider."""
    assert _LLM_CLIENT_CLASSES[LlmProvider.DEEPSEEK] is DeepSeekNonThinkingClient
    assert _LLM_CLIENT_CLASSES[LlmProvider.OPENAI] is OpenAIGenericClient


def test_llm_provider_parse_accepts_surrounding_whitespace_and_casing() -> None:
    """Provider values are normalized, so "DeepSeek" and " deepseek " both resolve."""
    assert LlmProvider.parse("DeepSeek") is LlmProvider.DEEPSEEK
    assert LlmProvider.parse("  deepseek  ") is LlmProvider.DEEPSEEK


def test_llm_provider_parse_rejects_an_unsupported_value() -> None:
    """A typo'd or unsupported provider fails loudly instead of silently using a default."""
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        LlmProvider.parse("deepsek")


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
