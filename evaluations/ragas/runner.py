"""Run a Ragas evaluation over the sample set and reduce it to a report."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluations.core.dataset import EvalSample
from evaluations.core.report import EvaluationReport


def run_evaluation(
    samples: Sequence[EvalSample],
    metrics: list[Any],
    *,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> EvaluationReport:
    """Evaluate the samples with Ragas and aggregate the per-sample scores into a report."""
    from ragas import EvaluationDataset, evaluate

    dataset = EvaluationDataset.from_list([sample.to_ragas_record() for sample in samples])
    result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)
    return EvaluationReport.from_per_sample(result.scores)
