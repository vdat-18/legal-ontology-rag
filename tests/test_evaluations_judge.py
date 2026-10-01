"""Tests for flattening typed BAML judge evaluations into score mappings."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from legal_rag.evaluation.generation.core.dataset import EvalSample
from legal_rag.evaluation.generation.judge import _evaluation_item, _index_evaluations, _to_scores


def test_evaluation_item_carries_benchmark_context_into_judge_input() -> None:
    """History, expected answerability, and required claims are all sent to BAML."""
    item = _evaluation_item(
        7,
        EvalSample(
            user_input="Còn xe máy thì sao?",
            response="answer",
            retrieved_contexts=("context",),
            reference="reference",
            history=(("Ô tô thì sao?", "first answer"),),
            answerability="partially_answerable",
            required_claims=("claim one", "claim two"),
            query_date=date(2025, 1, 1),
            query_dates=(date(2025, 1, 1), date(2026, 3, 1)),
            reference_date=date(2026, 8, 31),
        ),
    )

    assert item.sample_id == "7"
    assert [(turn.question, turn.answer) for turn in item.history] == [("Ô tô thì sao?", "first answer")]
    assert item.answerability == "partially_answerable"
    assert item.required_claims == ["claim one", "claim two"]
    assert item.query_date == "2025-01-01"
    assert item.query_dates == ["2025-01-01", "2026-03-01"]
    assert item.reference_date == "2026-08-31"


def test_judge_result_index_rejects_duplicate_extra_evaluation() -> None:
    """A duplicate cannot be silently collapsed even when every expected ID is present."""
    items = [SimpleNamespace(sample_id="0"), SimpleNamespace(sample_id="1")]
    evaluations = [
        SimpleNamespace(sample_id="0"),
        SimpleNamespace(sample_id="1"),
        SimpleNamespace(sample_id="1"),
    ]

    with pytest.raises(ValueError, match="exactly one evaluation"):
        _index_evaluations(evaluations, items)


def test_to_scores_flattens_a_complete_evaluation() -> None:
    """Every metric maps to a float, with temporal true becoming 1.0."""
    evaluation = SimpleNamespace(
        faithfulness=0.75,
        correctness=4,
        answer_relevancy=5,
        temporal_correctness=True,
        answerability_correctness=True,
        required_claim_coverage=0.5,
    )

    assert _to_scores(evaluation) == {
        "faithfulness": 0.75,
        "correctness": 4.0,
        "answer_relevancy": 5.0,
        "temporal_correctness": 1.0,
        "answerability_correctness": 1.0,
        "required_claim_coverage": 0.5,
    }


def test_to_scores_maps_abstaining_temporal_to_none() -> None:
    """A null temporal judgment (abstaining answer) stays None instead of forcing 0.0."""
    evaluation = SimpleNamespace(
        faithfulness=1.0,
        correctness=1,
        answer_relevancy=2,
        temporal_correctness=None,
        answerability_correctness=None,
        required_claim_coverage=None,
    )

    assert _to_scores(evaluation)["temporal_correctness"] is None


def test_to_scores_maps_false_temporal_to_zero() -> None:
    """An explicitly wrong law version still scores 0.0."""
    evaluation = SimpleNamespace(
        faithfulness=1.0,
        correctness=3,
        answer_relevancy=4,
        temporal_correctness=False,
        answerability_correctness=False,
        required_claim_coverage=1.0,
    )

    assert _to_scores(evaluation)["temporal_correctness"] == 0.0
    assert _to_scores(evaluation)["answerability_correctness"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("faithfulness", -0.01),
        ("faithfulness", 1.01),
        ("correctness", 0),
        ("correctness", 6),
        ("answer_relevancy", float("nan")),
        ("required_claim_coverage", 1.1),
    ],
)
def test_to_scores_rejects_out_of_range_or_non_finite_numeric_output(field: str, value: float) -> None:
    """Malformed judge numbers fail loudly instead of contaminating aggregates."""
    evaluation = SimpleNamespace(
        faithfulness=1.0,
        correctness=5,
        answer_relevancy=5,
        temporal_correctness=True,
        answerability_correctness=True,
        required_claim_coverage=1.0,
    )
    setattr(evaluation, field, value)

    with pytest.raises(ValueError, match=field):
        _to_scores(evaluation)


@pytest.mark.parametrize("field", ["temporal_correctness", "answerability_correctness"])
def test_to_scores_rejects_non_boolean_binary_output(field: str) -> None:
    """Binary judgments accept only typed booleans or null."""
    evaluation = SimpleNamespace(
        faithfulness=1.0,
        correctness=5,
        answer_relevancy=5,
        temporal_correctness=True,
        answerability_correctness=True,
        required_claim_coverage=1.0,
    )
    setattr(evaluation, field, 1)

    with pytest.raises(TypeError, match=field):
        _to_scores(evaluation)
