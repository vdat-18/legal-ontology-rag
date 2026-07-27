"""Tests for collecting configured LLM API keys for concurrent Graphiti indexing."""

from __future__ import annotations

from legal_rag.graphiti_rag.client import load_llm_api_keys


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
