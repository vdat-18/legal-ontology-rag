"""Tests for adaptive RAG environment settings."""

from __future__ import annotations

import pytest

from legal_rag.settings import AdaptiveRagSettings


def test_adaptive_settings_defaults():
    """Defaults select bounded hybrid reranking and deterministic RRF."""
    settings = AdaptiveRagSettings()

    assert settings.base_search_mode == "hybrid_rerank"
    assert settings.max_subqueries == 3
    assert settings.per_leg_limit == 10
    assert settings.rrf_k == 60


def test_adaptive_settings_load_environment_overrides():
    """Every adaptive tunable is configurable through its declared prefix."""
    settings = AdaptiveRagSettings.from_environment(
        {
            "ADAPTIVE_RAG_BASE_SEARCH_MODE": "hierarchical",
            "ADAPTIVE_RAG_MAX_SUBQUERIES": "4",
            "ADAPTIVE_RAG_PER_LEG_LIMIT": "12",
            "ADAPTIVE_RAG_RRF_K": "30",
            "ADAPTIVE_RAG_PLANNER_MAX_OUTPUT_TOKENS": "1024",
            "ADAPTIVE_RAG_PLANNER_TEMPERATURE": "0.2",
            "ADAPTIVE_RAG_PLANNER_MAX_RESPONSE_BYTES": "131072",
        }
    )

    assert settings == AdaptiveRagSettings(
        base_search_mode="hierarchical",
        max_subqueries=4,
        per_leg_limit=12,
        rrf_k=30,
        planner_max_output_tokens=1024,
        planner_temperature=0.2,
        planner_max_response_bytes=131_072,
    )


@pytest.mark.parametrize("base_search_mode", ["", "adaptive", "unknown"])
def test_adaptive_settings_reject_invalid_base_mode(base_search_mode):
    """Adaptive orchestration cannot select an unsupported or recursive base mode."""
    with pytest.raises(ValueError, match="base_search_mode"):
        AdaptiveRagSettings(base_search_mode=base_search_mode)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_subqueries", 1),
        ("per_leg_limit", 0),
        ("rrf_k", 0),
        ("planner_max_output_tokens", 0),
        ("planner_max_response_bytes", 0),
        ("planner_temperature", -0.1),
        ("planner_temperature", 2.1),
    ],
)
def test_adaptive_settings_reject_invalid_limits(field, value):
    """Invalid external planning limits fail before model or database calls."""
    with pytest.raises(ValueError, match=field):
        AdaptiveRagSettings(**{field: value})
