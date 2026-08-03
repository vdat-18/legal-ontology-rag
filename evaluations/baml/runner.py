"""Judge samples in batches, running the batches concurrently with bounded parallelism."""

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
    batch_size: int,
    concurrency: int,
) -> EvaluationReport:
    """Score all *samples* in batches of *batch_size*, at most *concurrency* batches at once.

    Each batch is one judge call (many samples scored together); a semaphore bounds how many batch
    calls run at the same time so the shared LLM provider is not overwhelmed. Results keep the
    input order of the samples.
    """
    batches = [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]
    semaphore = asyncio.Semaphore(concurrency)

    async def _score_batch(batch: Sequence[EvalSample]) -> list[dict[str, float | None]]:
        async with semaphore:
            return await judge.score_batch(batch)

    batch_scores = await asyncio.gather(*(_score_batch(batch) for batch in batches))
    per_sample = [scores for batch in batch_scores for scores in batch]
    return EvaluationReport.from_per_sample(per_sample)
