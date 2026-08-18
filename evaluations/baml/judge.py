"""LLM judge that scores a batch of answers in one BAML call through the project's runtime.

One ``EvaluateAnswers`` call scores a whole batch of samples, and each result carries every metric
for its sample as a typed object (the QUEST 1-5 rubric, faithfulness, answer relevancy, temporal
correctness, and optional factual correctness). So a batch of samples costs a single LLM call, and
the runner runs many batches concurrently.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from evaluations.core.dataset import EvalSample
from evaluations.quest.rubric import QUEST_DIMENSIONS, TEMPORAL_CORRECTNESS_NAME

if TYPE_CHECKING:
    from legal_rag.settings import Settings

_JUDGE_MAX_OUTPUT_TOKENS = 4096
_JUDGE_MAX_RESPONSE_BYTES = 1_048_576


class JudgeClient(Protocol):
    """Scores a batch of samples into one metric-name to score mapping per sample, in input order."""

    async def score_batch(self, samples: Sequence[EvalSample]) -> list[dict[str, float | None]]:
        """Return one metric-name to score mapping for each sample in *samples*, in order."""
        ...


class BamlJudgeClient:
    """Judge backed by the shared BAML runtime; one bounded call scores a whole batch of samples."""

    def __init__(self, settings: Settings) -> None:
        """Pin one BAML runtime provider and the judge call's execution bounds."""
        from legal_rag.baml_transport import BamlCaller

        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            max_response_bytes=_JUDGE_MAX_RESPONSE_BYTES,
        )

    async def score_batch(self, samples: Sequence[EvalSample]) -> list[dict[str, float | None]]:
        """Run one BAML judgment over the whole batch and return scores in the samples' order."""
        from legal_rag.baml_client.sync_client import b
        from legal_rag.baml_client.types import LegalAnswerToEvaluate

        items = [
            LegalAnswerToEvaluate(
                sample_id=str(index),
                question=sample.user_input,
                answer=sample.response,
                contexts=list(sample.retrieved_contexts),
                reference=sample.reference,
            )
            for index, sample in enumerate(samples)
        ]
        request = b.request.EvaluateAnswers(items, baml_options=self._caller.options)
        raw_output = await self._caller.execute_async(request)
        evaluations = b.parse.EvaluateAnswers(raw_output, baml_options=self._caller.options)

        evaluations_by_id = {evaluation.sample_id: evaluation for evaluation in evaluations}
        if len(evaluations_by_id) != len(items) or set(evaluations_by_id) != {item.sample_id for item in items}:
            msg = "Judge response must contain exactly one evaluation for every sample in the batch"
            raise ValueError(msg)
        return [_to_scores(evaluations_by_id[item.sample_id]) for item in items]


def _to_scores(evaluation: Any) -> dict[str, float | None]:
    """Flatten one typed BAML evaluation into a metric-name to score mapping."""
    scores: dict[str, float | None] = {
        f"quest_{dimension}": float(getattr(evaluation, dimension)) for dimension in QUEST_DIMENSIONS
    }
    scores["faithfulness"] = float(evaluation.faithfulness)
    scores["answer_relevancy"] = float(evaluation.answer_relevancy)
    scores[TEMPORAL_CORRECTNESS_NAME] = 1.0 if evaluation.temporal_correctness else 0.0
    scores["factual_correctness"] = (
        float(evaluation.factual_correctness) if evaluation.factual_correctness is not None else None
    )
    return scores
