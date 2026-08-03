"""Score every sample through the judge concurrently, with bounded parallelism."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from evaluations.core.report import EvaluationReport

if TYPE_CHECKING:
    from evaluations.baml.judge import JudgeClient
    from evaluations.core.dataset import EvalSample


async def evaluate_samples(
    samples: Sequence[EvalSample],
    judge: JudgeClient,
    *,
    concurrency: int,
) -> EvaluationReport:
    """Judge all *samples* with at most *concurrency* calls in flight, then aggregate the scores.

    Each sample is one batched judge call (all metrics at once); a semaphore bounds how many run
    at the same time so the shared LLM provider is not overwhelmed.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _score(sample: EvalSample) -> dict[str, float | None]:
        async with semaphore:
            return await judge.score(sample)

    per_sample = await asyncio.gather(*(_score(sample) for sample in samples))
    return EvaluationReport.from_per_sample(per_sample)
