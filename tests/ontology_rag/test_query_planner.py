"""Tests for typed ontology query planning."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from legal_rag.ontology_rag.query_planner import (
    BamlOntologyQueryPlanner,
    TemporalQueryMode,
    normalize_query_plan,
)
from legal_rag.settings import Settings


def _settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=30,
    )


def test_baml_planner_parses_comparison_dates() -> None:
    """Typed output preserves a standalone query and ordered comparison dates."""
    planner = BamlOntologyQueryPlanner(_settings())
    planner._caller.execute = MagicMock(
        return_value=(
            '{"query_vi":"So sánh quy định cũ và mới",'
            '"temporal_mode":"CompareDates",'
            '"query_dates":["2026-03-01","2026-08-15"]}'
        )
    )

    result = planner.plan("So sánh", (), date(2026, 10, 1))

    assert result.search_query == "So sánh quy định cũ và mới"
    assert result.temporal_mode == TemporalQueryMode.COMPARE_DATES
    assert result.query_dates == (date(2026, 3, 1), date(2026, 8, 15))


def test_normalize_current_plan_rejects_invented_dates() -> None:
    """A current query cannot silently become a historical query."""
    with pytest.raises(ValueError, match="current ontology plans require 0 to 0"):
        normalize_query_plan("Mức phạt hiện nay", TemporalQueryMode.CURRENT, ["2026-10-01"])


def test_normalize_at_date_rejects_invalid_iso_date() -> None:
    """Invalid model output fails at the typed system boundary."""
    with pytest.raises(ValueError, match="invalid ISO date"):
        normalize_query_plan("Mức phạt", TemporalQueryMode.AT_DATE, ["01/03/2026"])


def test_normalize_at_date_rejects_noncanonical_iso_date() -> None:
    """Basic ISO dates cannot bypass the canonical YYYY-MM-DD contract."""
    with pytest.raises(ValueError, match="invalid ISO date"):
        normalize_query_plan("Mức phạt", TemporalQueryMode.AT_DATE, ["20260301"])


def test_normalize_comparison_deduplicates_dates_without_reordering() -> None:
    """Repeated dates contribute one temporal target while source order is stable."""
    result = normalize_query_plan(
        "  So sánh   quy định  ",
        TemporalQueryMode.COMPARE_DATES,
        ["2025-01-01", "2025-01-01", "2026-03-01"],
    )

    assert result.search_query == "So sánh quy định"
    assert result.query_dates == (date(2025, 1, 1), date(2026, 3, 1))
