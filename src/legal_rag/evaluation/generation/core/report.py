"""Aggregate per-sample generation scores into a typed, JSON-serializable report."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

GENERATION_METRIC_NAMES = (
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "temporal_correctness",
    "answerability_correctness",
    "required_claim_coverage",
)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Per-sample metric scores plus their means across the dataset."""

    sample_count: int
    aggregate: dict[str, float | None]
    metric_coverage: dict[str, dict[str, int | float]]
    per_sample: tuple[dict[str, Any], ...]

    @classmethod
    def from_per_sample(cls, per_sample: Sequence[Mapping[str, Any]]) -> EvaluationReport:
        """Build a report from per-sample scores, averaging each metric over valid values."""
        rows = tuple(_normalize_row(row, index=index) for index, row in enumerate(per_sample, 1))
        aggregate = {name: _mean(_finite(row.get(name)) for row in rows) for name in sorted(GENERATION_METRIC_NAMES)}
        metric_coverage = {
            name: _coverage(_finite(row.get(name)) for row in rows) for name in sorted(GENERATION_METRIC_NAMES)
        }
        return cls(
            sample_count=len(rows),
            aggregate=aggregate,
            metric_coverage=metric_coverage,
            per_sample=rows,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the report."""
        return {
            "sample_count": self.sample_count,
            "aggregate": self.aggregate,
            "metric_coverage": self.metric_coverage,
            "per_sample": [dict(row) for row in self.per_sample],
        }


def _finite(value: Any) -> float | None:
    """Return *value* as a float, or None when it is missing or not a finite number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _normalize_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    """Preserve stable sample identity while normalizing metric values."""
    missing = [name for name in GENERATION_METRIC_NAMES if name not in row]
    if missing:
        msg = f"Generation score row {index} is missing required metrics: {', '.join(missing)}"
        raise ValueError(msg)
    unexpected = sorted(name for name in row if name != "sample_id" and name not in GENERATION_METRIC_NAMES)
    if unexpected:
        msg = f"Generation score row {index} contains unexpected metrics: {', '.join(unexpected)}"
        raise ValueError(msg)
    normalized: dict[str, Any] = {}
    for name, value in row.items():
        if name == "sample_id" and isinstance(value, str):
            normalized[name] = value
        else:
            normalized[name] = _finite(value)
    return normalized


def _mean(values: Iterable[float | None]) -> float | None:
    """Return the mean of the non-None values, or None when there are none."""
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _coverage(values: Iterable[float | None]) -> dict[str, int | float]:
    """Report how many samples contributed to a metric's aggregate."""
    scores = tuple(values)
    scored_count = sum(value is not None for value in scores)
    sample_count = len(scores)
    return {
        "scored_count": scored_count,
        "null_count": sample_count - scored_count,
        "coverage": scored_count / sample_count if sample_count else 0.0,
    }
