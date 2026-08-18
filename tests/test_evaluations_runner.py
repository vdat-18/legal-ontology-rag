"""Tests for the batched, concurrent evaluation runner (with a fake judge)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from evaluations.baml.runner import evaluate_samples
from evaluations.core.dataset import EvalSample


class _RecordingJudge:
    """Return preset per-question scores while tracking batch sizes and peak batch concurrency."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.active = 0
        self.max_active = 0
        self.batch_sizes: list[int] = []

    async def score_batch(self, samples: Sequence[EvalSample]) -> list[dict[str, float | None]]:
        """Record the batch, yield once so sibling batches interleave, and return preset scores."""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.batch_sizes.append(len(samples))
        await asyncio.sleep(0)
        self.active -= 1
        return [{"faithfulness": self._scores[sample.user_input]} for sample in samples]


def _sample(user_input: str) -> EvalSample:
    """Build one minimal evaluation sample."""
    return EvalSample(user_input=user_input, response="answer")


def test_evaluate_samples_splits_into_batches_of_the_requested_size() -> None:
    """Samples are grouped into batches of batch_size, with a smaller final batch."""
    samples = [_sample(f"q{index}") for index in range(5)]
    judge = _RecordingJudge({sample.user_input: 1.0 for sample in samples})

    report = asyncio.run(evaluate_samples(samples, judge, batch_size=2, concurrency=8))

    assert sorted(judge.batch_sizes) == [1, 2, 2]
    assert report.sample_count == 5


def test_evaluate_samples_bounds_concurrent_batches() -> None:
    """No more than *concurrency* batch calls run at once."""
    samples = [_sample(f"q{index}") for index in range(6)]
    judge = _RecordingJudge({sample.user_input: 1.0 for sample in samples})

    asyncio.run(evaluate_samples(samples, judge, batch_size=1, concurrency=2))

    assert judge.batch_sizes == [1, 1, 1, 1, 1, 1]
    assert judge.max_active == 2


def test_evaluate_samples_preserves_sample_order_across_batches() -> None:
    """Per-sample scores keep the original sample order even though batches run concurrently."""
    samples = [_sample("q0"), _sample("q1"), _sample("q2")]
    judge = _RecordingJudge({"q0": 0.0, "q1": 0.5, "q2": 1.0})

    report = asyncio.run(evaluate_samples(samples, judge, batch_size=1, concurrency=3))

    assert [row["faithfulness"] for row in report.per_sample] == [0.0, 0.5, 1.0]
    assert report.aggregate == {"faithfulness": 0.5}
