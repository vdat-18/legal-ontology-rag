"""LLM judge that scores one answer per sample through the project's BAML runtime.

One ``EvaluateLegalAnswer`` call returns every metric for a sample as a typed object (the QUEST
1-5 rubric, faithfulness, answer relevancy, temporal correctness, and optional factual
correctness), so a sample costs a single LLM call instead of one call per metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from evaluations.core.dataset import EvalSample
from evaluations.quest.rubric import QUEST_DIMENSIONS, TEMPORAL_CORRECTNESS_NAME

if TYPE_CHECKING:
    from legal_rag.settings import Settings

_JUDGE_MAX_OUTPUT_TOKENS = 1024
_JUDGE_MAX_RESPONSE_BYTES = 1_048_576


class JudgeClient(Protocol):
    """Scores one evaluation sample into a mapping of metric name to numeric score."""

    async def score(self, sample: EvalSample) -> dict[str, float | None]:
        """Return one metric-name to score mapping for *sample*."""
        ...


class BamlJudgeClient:
    """Judge backed by the shared BAML runtime; one bounded typed call scores all metrics at once."""

    def __init__(self, settings: Settings) -> None:
        """Pin one BAML runtime provider and the judge call's execution bounds."""
        from legal_rag.baml_transport import BamlCaller

        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            max_response_bytes=_JUDGE_MAX_RESPONSE_BYTES,
        )

    async def score(self, sample: EvalSample) -> dict[str, float | None]:
        """Run one BAML judgment for *sample* and flatten it into metric scores."""
        from legal_rag.baml_client.sync_client import b

        request = b.request.EvaluateLegalAnswer(
            question=sample.user_input,
            answer=sample.response,
            contexts=list(sample.retrieved_contexts),
            reference=sample.reference,
            baml_options=self._caller.options,
        )
        raw_output = await self._caller.execute_async(request)
        evaluation = b.parse.EvaluateLegalAnswer(raw_output, baml_options=self._caller.options)
        return _to_scores(evaluation)


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
