"""Tests for the concurrent, batched evaluation runner (with a fake judge)."""

from __future__ import annotations

import asyncio

from evaluations.baml.runner import evaluate_samples
from evaluations.core.dataset import EvalSample


class _RecordingJudge:
    """Return preset per-question scores while tracking peak concurrency."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.active = 0
        self.max_active = 0

    async def score(self, sample: EvalSample) -> dict[str, float | None]:
        """Record concurrency, yield once so siblings interleave, and return the preset score."""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return {"faithfulness": self._scores[sample.user_input]}


def _sample(user_input: str) -> EvalSample:
    """Build one minimal evaluation sample."""
    return EvalSample(user_input=user_input, response="answer")


def test_evaluate_samples_bounds_concurrency_and_scores_all() -> None:
    """No more than *concurrency* judgments run at once, and every sample is scored."""
    samples = [_sample(f"q{index}") for index in range(4)]
    judge = _RecordingJudge({sample.user_input: 1.0 for sample in samples})

    report = asyncio.run(evaluate_samples(samples, judge, concurrency=2))

    assert judge.max_active == 2
    assert report.sample_count == 4


def test_evaluate_samples_runs_serially_at_concurrency_one() -> None:
    """A concurrency of one judges strictly one sample at a time."""
    samples = [_sample("q0"), _sample("q1")]
    judge = _RecordingJudge({"q0": 1.0, "q1": 1.0})

    asyncio.run(evaluate_samples(samples, judge, concurrency=1))

    assert judge.max_active == 1


def test_evaluate_samples_aggregates_per_metric_means() -> None:
    """The report averages each metric across the judged samples."""
    samples = [_sample("q0"), _sample("q1")]
    judge = _RecordingJudge({"q0": 1.0, "q1": 0.0})

    report = asyncio.run(evaluate_samples(samples, judge, concurrency=2))

    assert report.aggregate == {"faithfulness": 0.5}
