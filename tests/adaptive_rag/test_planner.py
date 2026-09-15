"""Tests for adaptive query planning and validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_rag.adaptive_rag.models import AdaptiveRoute
from legal_rag.adaptive_rag.planner import BamlAdaptivePlanner, _normalize_plan
from legal_rag.settings import AdaptiveRagSettings, Settings


def _settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=30,
    )


def test_baml_planner_parses_a_direct_plan():
    """A typed BAML response becomes the internal immutable plan."""
    planner = BamlAdaptivePlanner(_settings(), AdaptiveRagSettings())
    planner._caller.execute = MagicMock(return_value='{"route":"Direct","subqueries":[]}')

    result = planner.plan("Mức phạt vượt đèn đỏ là bao nhiêu?")

    assert result.route == AdaptiveRoute.DIRECT
    assert result.subqueries == ()


def test_normalize_multi_hop_plan_deduplicates_queries_and_preserves_order():
    """Whitespace, case, and Unicode-equivalent duplicates contribute one retrieval leg."""
    result = _normalize_plan(
        AdaptiveRoute.MULTI_HOP,
        ["  Nghĩa vụ dừng xe?  ", "nghĩa vụ dừng xe?", "Mức xử phạt là bao nhiêu?"],
        "Câu hỏi gốc",
        3,
    )

    assert result.subqueries == ("Nghĩa vụ dừng xe?", "Mức xử phạt là bao nhiêu?")


def test_normalize_multi_hop_plan_removes_the_original_query():
    """The mandatory original-query leg is not repeated as a subquery leg."""
    result = _normalize_plan(
        AdaptiveRoute.MULTI_HOP,
        ["Câu hỏi gốc", "Căn cứ thứ nhất?", "Căn cứ thứ hai?"],
        "  câu hỏi gốc ",
        3,
    )

    assert result.subqueries == ("Căn cứ thứ nhất?", "Căn cứ thứ hai?")


def test_direct_plan_ignores_subqueries():
    """Direct routing stays usable when the model emits irrelevant subqueries."""
    result = _normalize_plan(AdaptiveRoute.DIRECT, ["extra"], "question", 3)

    assert result.route == AdaptiveRoute.DIRECT
    assert result.subqueries == ()


@pytest.mark.parametrize(
    "subqueries",
    [[], ["only one"], ["one", "two", "three", "four"]],
)
def test_multi_hop_plan_rejects_an_invalid_unique_subquery_count(subqueries):
    """Multi-hop plans stay within the configured two-to-maximum bound."""
    with pytest.raises(ValueError, match="must contain 2 to 3"):
        _normalize_plan(AdaptiveRoute.MULTI_HOP, subqueries, "question", 3)


def test_multi_hop_plan_rejects_empty_subqueries():
    """Empty planner strings fail clearly instead of becoming silent no-op legs."""
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize_plan(AdaptiveRoute.MULTI_HOP, ["first", "   "], "question", 3)


def test_planner_transport_errors_propagate():
    """Provider failures remain visible to evaluation manifest handling."""
    planner = BamlAdaptivePlanner(_settings(), AdaptiveRagSettings())
    planner._caller.execute = MagicMock(side_effect=RuntimeError("provider failed"))

    with pytest.raises(RuntimeError, match="provider failed"):
        planner.plan("question")
