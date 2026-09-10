"""Tests for flattening typed BAML judge evaluations into score mappings."""

from __future__ import annotations

from types import SimpleNamespace

from legal_rag.evaluation.generation.judge import _to_scores


def test_to_scores_flattens_a_complete_evaluation() -> None:
    """Every metric maps to a float, with temporal true becoming 1.0."""
    evaluation = SimpleNamespace(
        faithfulness=0.75,
        correctness=4,
        answer_relevancy=5,
        temporal_correctness=True,
    )

    assert _to_scores(evaluation) == {
        "faithfulness": 0.75,
        "correctness": 4.0,
        "answer_relevancy": 5.0,
        "temporal_correctness": 1.0,
    }


def test_to_scores_maps_abstaining_temporal_to_none() -> None:
    """A null temporal judgment (abstaining answer) stays None instead of forcing 0.0."""
    evaluation = SimpleNamespace(
        faithfulness=1.0,
        correctness=1,
        answer_relevancy=2,
        temporal_correctness=None,
    )

    assert _to_scores(evaluation)["temporal_correctness"] is None


def test_to_scores_maps_false_temporal_to_zero() -> None:
    """An explicitly wrong law version still scores 0.0."""
    evaluation = SimpleNamespace(
        faithfulness=1.0,
        correctness=3,
        answer_relevancy=4,
        temporal_correctness=False,
    )

    assert _to_scores(evaluation)["temporal_correctness"] == 0.0
