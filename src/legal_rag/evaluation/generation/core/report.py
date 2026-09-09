"""Aggregate per-sample generation scores into a typed, JSON-serializable report."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Per-sample metric scores plus their means across the dataset."""

    sample_count: int
    aggregate: dict[str, float | None]
    per_sample: tuple[dict[str, float | None], ...]

    @classmethod
    def from_per_sample(cls, per_sample: Sequence[Mapping[str, Any]]) -> EvaluationReport:
        """Build a report from per-sample scores, averaging each metric over valid values."""
        rows = tuple({name: _finite(value) for name, value in row.items()} for row in per_sample)
        metric_names = {name for row in rows for name in row}
        aggregate = {name: _mean(row.get(name) for row in rows) for name in sorted(metric_names)}
        return cls(sample_count=len(rows), aggregate=aggregate, per_sample=rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the report."""
        return {
            "sample_count": self.sample_count,
            "aggregate": self.aggregate,
            "per_sample": [dict(row) for row in self.per_sample],
        }


def _finite(value: Any) -> float | None:
    """Return *value* as a float, or None when it is missing or not a finite number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float | None]) -> float | None:
    """Return the mean of the non-None values, or None when there are none."""
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
