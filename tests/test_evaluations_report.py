"""Tests for aggregating Ragas per-sample scores into a report."""

from __future__ import annotations

import math

from evaluations.core.report import EvaluationReport


def test_from_per_sample_averages_each_metric_over_valid_values() -> None:
    """Each metric's aggregate is the mean of its finite per-sample scores."""
    report = EvaluationReport.from_per_sample(
        [
            {"faithfulness": 1.0, "answer_relevancy": 0.5},
            {"faithfulness": 0.0, "answer_relevancy": 0.9},
        ]
    )

    assert report.sample_count == 2
    assert report.aggregate == {"answer_relevancy": 0.7, "faithfulness": 0.5}


def test_from_per_sample_ignores_missing_and_non_finite_scores() -> None:
    """None and NaN scores are dropped from both the row and the mean; all-missing yields None."""
    report = EvaluationReport.from_per_sample(
        [
            {"faithfulness": 1.0, "context_recall": None},
            {"faithfulness": math.nan, "context_recall": None},
        ]
    )

    assert report.aggregate == {"context_recall": None, "faithfulness": 1.0}
    assert report.per_sample[1]["faithfulness"] is None


def test_to_dict_is_json_serializable_shape() -> None:
    """The report renders to a plain aggregate + per-sample structure."""
    report = EvaluationReport.from_per_sample([{"faithfulness": 1.0}])

    assert report.to_dict() == {
        "sample_count": 1,
        "aggregate": {"faithfulness": 1.0},
        "per_sample": [{"faithfulness": 1.0}],
    }
