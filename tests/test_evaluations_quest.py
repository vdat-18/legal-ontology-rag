"""Tests for the QUEST rubric definitions used by the LLM-judge metrics."""

from __future__ import annotations

from evaluations.quest.rubric import (
    QUEST_RUBRICS,
    TEMPORAL_CORRECTNESS_DEFINITION,
    TEMPORAL_CORRECTNESS_NAME,
)

_EXPECTED_DIMENSIONS = frozenset(
    {
        "quality_of_information",
        "understanding_and_reasoning",
        "expression_style",
        "safety_and_harm",
        "trust_and_confidence",
    }
)
_EXPECTED_SCORE_KEYS = frozenset(f"score{level}_description" for level in range(1, 6))


def test_quest_covers_the_five_principles() -> None:
    """The rubric encodes exactly the five QUEST evaluation principles."""
    assert set(QUEST_RUBRICS) == _EXPECTED_DIMENSIONS


def test_each_dimension_defines_a_full_one_to_five_scale() -> None:
    """Every principle carries a non-empty description for each score from 1 to 5."""
    for rubric in QUEST_RUBRICS.values():
        assert set(rubric) == _EXPECTED_SCORE_KEYS
        assert all(description.strip() for description in rubric.values())


def test_temporal_correctness_aspect_is_defined() -> None:
    """The temporal-correctness aspect has a stable name and a non-empty definition."""
    assert TEMPORAL_CORRECTNESS_NAME == "temporal_correctness"
    assert TEMPORAL_CORRECTNESS_DEFINITION.strip()
