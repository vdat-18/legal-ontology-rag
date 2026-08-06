"""Tests for the QUEST metric registry used to label the judge output."""

from __future__ import annotations

from evaluations.quest.rubric import (
    QUEST_DESCRIPTIONS,
    QUEST_DIMENSIONS,
    TEMPORAL_CORRECTNESS_NAME,
)

_EXPECTED_DIMENSIONS = (
    "quality_of_information",
    "understanding_and_reasoning",
    "expression_style",
    "safety_and_harm",
    "trust_and_confidence",
)


def test_quest_registers_the_five_principles_in_order() -> None:
    """The registry lists exactly the five QUEST evaluation principles."""
    assert QUEST_DIMENSIONS == _EXPECTED_DIMENSIONS


def test_every_dimension_has_a_one_line_description() -> None:
    """Each principle carries a non-empty human-readable description for reports."""
    assert set(QUEST_DESCRIPTIONS) == set(_EXPECTED_DIMENSIONS)
    assert all(description.strip() for description in QUEST_DESCRIPTIONS.values())


def test_temporal_correctness_name_is_stable() -> None:
    """The temporal-correctness metric keeps a stable name."""
    assert TEMPORAL_CORRECTNESS_NAME == "temporal_correctness"
