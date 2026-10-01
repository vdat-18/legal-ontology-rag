"""Tests for aggregating per-sample generation scores into a report."""

from __future__ import annotations

import math
from typing import Any

import pytest

from legal_rag.evaluation.generation.core.report import GENERATION_METRIC_NAMES, EvaluationReport


def _scores(**overrides: Any) -> dict[str, Any]:
    """Return one complete six-metric judge result."""
    scores: dict[str, Any] = {
        "faithfulness": 1.0,
        "correctness": 5.0,
        "answer_relevancy": 4.0,
        "temporal_correctness": 1.0,
        "answerability_correctness": 1.0,
        "required_claim_coverage": 1.0,
    }
    scores.update(overrides)
    return scores


def test_from_per_sample_averages_each_metric_over_valid_values() -> None:
    """Each metric's aggregate is the mean of its finite per-sample scores."""
    report = EvaluationReport.from_per_sample(
        [
            _scores(faithfulness=1.0, answer_relevancy=4.0),
            _scores(faithfulness=0.0, answer_relevancy=5.0),
        ]
    )

    assert report.sample_count == 2
    assert report.aggregate == {
        "answer_relevancy": 4.5,
        "answerability_correctness": 1.0,
        "correctness": 5.0,
        "faithfulness": 0.5,
        "required_claim_coverage": 1.0,
        "temporal_correctness": 1.0,
    }
    assert report.metric_coverage == {
        name: {"scored_count": 2, "null_count": 0, "coverage": 1.0} for name in GENERATION_METRIC_NAMES
    }


def test_from_per_sample_ignores_null_and_non_finite_scores() -> None:
    """None and NaN scores are omitted from means while coverage exposes the missing values."""
    report = EvaluationReport.from_per_sample(
        [
            _scores(faithfulness=1.0, temporal_correctness=None),
            _scores(faithfulness=math.nan, temporal_correctness=None),
        ]
    )

    assert report.aggregate["faithfulness"] == 1.0
    assert report.aggregate["temporal_correctness"] is None
    assert report.per_sample[1]["faithfulness"] is None
    assert report.metric_coverage["faithfulness"] == {
        "scored_count": 1,
        "null_count": 1,
        "coverage": 0.5,
    }
    assert report.metric_coverage["temporal_correctness"] == {
        "scored_count": 0,
        "null_count": 2,
        "coverage": 0.0,
    }


def test_from_per_sample_exposes_sparse_temporal_coverage() -> None:
    """A high temporal mean cannot hide that only one sample received a temporal judgment."""
    report = EvaluationReport.from_per_sample(
        [
            _scores(temporal_correctness=1.0),
            _scores(temporal_correctness=None),
            _scores(temporal_correctness=None),
        ]
    )

    assert report.aggregate["temporal_correctness"] == 1.0
    assert report.metric_coverage["temporal_correctness"] == {
        "scored_count": 1,
        "null_count": 2,
        "coverage": 1 / 3,
    }


def test_from_per_sample_rejects_an_incomplete_metric_row() -> None:
    """A judge result cannot silently omit one of the six versioned metrics."""
    incomplete = _scores()
    incomplete.pop("required_claim_coverage")

    with pytest.raises(ValueError, match="missing required metrics: required_claim_coverage"):
        EvaluationReport.from_per_sample([incomplete])


def test_empty_report_still_exposes_the_six_metric_contract() -> None:
    """An empty report retains stable aggregate and coverage keys."""
    report = EvaluationReport.from_per_sample([])

    assert report.aggregate == dict.fromkeys(sorted(GENERATION_METRIC_NAMES))
    assert report.metric_coverage == {
        name: {"scored_count": 0, "null_count": 0, "coverage": 0.0} for name in sorted(GENERATION_METRIC_NAMES)
    }


def test_to_dict_is_json_serializable_shape() -> None:
    """The report renders to a plain aggregate, coverage, and per-sample structure."""
    scores = _scores()
    report = EvaluationReport.from_per_sample([scores])

    assert report.to_dict() == {
        "sample_count": 1,
        "aggregate": {name: scores[name] for name in sorted(GENERATION_METRIC_NAMES)},
        "metric_coverage": {
            name: {"scored_count": 1, "null_count": 0, "coverage": 1.0} for name in sorted(GENERATION_METRIC_NAMES)
        },
        "per_sample": [scores],
    }


def test_from_per_sample_preserves_sample_id_without_aggregating_it() -> None:
    """Stable sample IDs stay in detail rows and never become numeric metrics."""
    scores = _scores(sample_id="question-1")
    report = EvaluationReport.from_per_sample([scores])

    assert set(report.aggregate) == set(GENERATION_METRIC_NAMES)
    assert report.per_sample == (scores,)
