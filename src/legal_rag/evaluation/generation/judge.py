"""LLM judge that scores a batch of answers in one BAML call through the project's runtime.

One ``EvaluateAnswers`` call scores a whole batch of samples, and each result carries every metric
for its sample as a typed object (faithfulness 0-1, correctness 1-5, answer_relevancy 1-5,
and temporal_correctness nullable binary — null when the answer abstains). So a batch of samples
costs a single LLM call, and the runner runs many batches concurrently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from legal_rag.evaluation.generation.core.dataset import EvalSample

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

        items = [_evaluation_item(index, sample) for index, sample in enumerate(samples)]
        request = b.request.EvaluateAnswers(items, baml_options=self._caller.options)
        raw_output = await self._caller.execute_async(request)
        evaluations = b.parse.EvaluateAnswers(raw_output, baml_options=self._caller.options)

        evaluations_by_id = _index_evaluations(evaluations, items)
        return [_to_scores(evaluations_by_id[item.sample_id]) for item in items]


def _index_evaluations(evaluations: Sequence[Any], items: Sequence[Any]) -> dict[str, Any]:
    """Require one and only one judge result for each requested sample identifier."""
    evaluations_by_id = {evaluation.sample_id: evaluation for evaluation in evaluations}
    expected_ids = {item.sample_id for item in items}
    if len(evaluations) != len(items) or len(evaluations_by_id) != len(items) or set(evaluations_by_id) != expected_ids:
        msg = "Judge response must contain exactly one evaluation for every sample in the batch"
        raise ValueError(msg)
    return evaluations_by_id


def _evaluation_item(index: int, sample: EvalSample) -> Any:
    """Map one repository sample onto the generated judge input contract."""
    from legal_rag.baml_client.types import ConversationTurn, LegalAnswerToEvaluate

    return LegalAnswerToEvaluate(
        sample_id=str(index),
        question=sample.user_input,
        answer=sample.response,
        contexts=list(sample.retrieved_contexts),
        reference=sample.reference,
        history=[ConversationTurn(question=question, answer=answer) for question, answer in sample.history],
        answerability=sample.answerability,
        required_claims=list(sample.required_claims),
        query_date=sample.query_date.isoformat() if sample.query_date is not None else None,
        query_dates=[item.isoformat() for item in sample.query_dates],
        reference_date=sample.reference_date.isoformat() if sample.reference_date is not None else None,
    )


def _to_scores(evaluation: Any) -> dict[str, float | None]:
    """Flatten and validate one typed BAML evaluation into bounded metric scores."""
    scores: dict[str, float | None] = {
        "faithfulness": _bounded_score(evaluation.faithfulness, name="faithfulness", minimum=0.0, maximum=1.0),
        "correctness": _bounded_score(evaluation.correctness, name="correctness", minimum=1.0, maximum=5.0),
        "answer_relevancy": _bounded_score(
            evaluation.answer_relevancy,
            name="answer_relevancy",
            minimum=1.0,
            maximum=5.0,
        ),
        "temporal_correctness": _optional_boolean_score(
            evaluation.temporal_correctness,
            name="temporal_correctness",
        ),
        "answerability_correctness": _optional_boolean_score(
            getattr(evaluation, "answerability_correctness", None),
            name="answerability_correctness",
        ),
        "required_claim_coverage": _optional_bounded_score(
            getattr(evaluation, "required_claim_coverage", None),
            name="required_claim_coverage",
            minimum=0.0,
            maximum=1.0,
        ),
    }
    return scores


def _bounded_score(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    """Return a finite numeric score inside its declared inclusive range."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"Judge metric {name} must be numeric, got {value!r}"
        raise TypeError(msg)
    score = float(value)
    if not math.isfinite(score) or not minimum <= score <= maximum:
        msg = f"Judge metric {name} must be between {minimum:g} and {maximum:g}, got {value!r}"
        raise ValueError(msg)
    return score


def _optional_bounded_score(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    """Validate an optional numeric judge score."""
    if value is None:
        return None
    return _bounded_score(value, name=name, minimum=minimum, maximum=maximum)


def _optional_boolean_score(value: Any, *, name: str) -> float | None:
    """Map an optional boolean judgment to 0/1 while rejecting malformed output."""
    if value is None:
        return None
    if not isinstance(value, bool):
        msg = f"Judge metric {name} must be true, false, or null, got {value!r}"
        raise TypeError(msg)
    return 1.0 if value else 0.0
