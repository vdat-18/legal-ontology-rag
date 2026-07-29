"""Offline metrics for Legal RAG evaluation datasets."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """One answer/retrieval evaluation sample."""

    question: str
    expected_answer: str | None = None
    actual_answer: str | None = None
    expected_chunk_ids: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    expected_source_names: tuple[str, ...] = ()
    retrieved_source_names: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EvaluationSample:
        """Create a sample from a flexible JSONL row."""
        result = _read_mapping(data, "result")
        graded_chunks = _read_sequence(data, "graded_chunks") or _read_sequence(result, "graded_chunks")
        contexts = _read_sequence(data, "contexts")
        expected_contexts = (
            _read_sequence(data, "expected_contexts")
            or _read_sequence(data, "reference_contexts")
            or _read_sequence(data, "ground_truth_contexts")
        )

        return cls(
            question=_read_first_string(data, ("question", "query", "user_input", "input")),
            expected_answer=_read_optional_string(data, ("expected_answer", "ground_truth", "ground_truth_answer")),
            actual_answer=_read_optional_string(data, ("actual_answer", "answer", "response"))
            or _read_optional_string(result, ("answer", "response")),
            expected_chunk_ids=_read_string_tuple(
                data,
                ("expected_chunk_ids", "reference_chunk_ids", "ground_truth_chunk_ids", "chunk_ids"),
            )
            or _extract_field_tuple(expected_contexts, ("chunk_id", "id")),
            retrieved_chunk_ids=_read_string_tuple(data, ("retrieved_chunk_ids", "predicted_chunk_ids"))
            or _extract_field_tuple(graded_chunks, ("chunk_id", "id"))
            or _extract_field_tuple(contexts, ("chunk_id", "id")),
            expected_source_names=_read_string_tuple(
                data,
                ("expected_source_names", "reference_source_names", "ground_truth_source_names", "source_names"),
            )
            or _extract_field_tuple(expected_contexts, ("source_name", "source", "name")),
            retrieved_source_names=_read_string_tuple(data, ("retrieved_source_names", "predicted_source_names"))
            or _extract_field_tuple(graded_chunks, ("source_name", "source", "name"))
            or _extract_field_tuple(contexts, ("source_name", "source", "name")),
        )


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate evaluation metrics."""

    sample_count: int
    retrieval_sample_count: int
    answer_sample_count: int
    k_values: tuple[int, ...]
    hit_rate: float | None
    hit_rate_at_k: dict[int, float | None]
    mrr: float | None
    precision_at_k: dict[int, float | None]
    recall_at_k: dict[int, float | None]
    ndcg_at_k: dict[int, float | None]
    context_precision: float | None
    context_recall: float | None
    answer_exact_match: float | None
    answer_token_f1: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "sample_count": self.sample_count,
            "retrieval_sample_count": self.retrieval_sample_count,
            "answer_sample_count": self.answer_sample_count,
            "k_values": list(self.k_values),
            "hit_rate": self.hit_rate,
            "hit_rate_at_k": _string_keyed_metrics(self.hit_rate_at_k),
            "mrr": self.mrr,
            "precision_at_k": _string_keyed_metrics(self.precision_at_k),
            "recall_at_k": _string_keyed_metrics(self.recall_at_k),
            "ndcg_at_k": _string_keyed_metrics(self.ndcg_at_k),
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_exact_match": self.answer_exact_match,
            "answer_token_f1": self.answer_token_f1,
        }


def load_jsonl_samples(path: Path) -> list[EvaluationSample]:
    """Load evaluation samples from a JSONL file."""
    samples: list[EvaluationSample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                msg = f"Expected JSON object at {path}:{line_number}"
                raise TypeError(msg)
            samples.append(EvaluationSample.from_mapping(row))
    return samples


def evaluate_samples(
    samples: Sequence[EvaluationSample],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> EvaluationMetrics:
    """Compute aggregate retrieval and answer metrics."""
    normalized_k_values = _normalize_k_values(k_values)
    retrieval_scores = [_retrieval_scores(sample, normalized_k_values) for sample in samples]
    retrieval_scores = [score for score in retrieval_scores if score is not None]

    answer_scores = [_answer_scores(sample) for sample in samples]
    answer_scores = [score for score in answer_scores if score is not None]

    return EvaluationMetrics(
        sample_count=len(samples),
        retrieval_sample_count=len(retrieval_scores),
        answer_sample_count=len(answer_scores),
        k_values=normalized_k_values,
        hit_rate=_mean(score["hit"] for score in retrieval_scores),
        hit_rate_at_k=_aggregate_at_k(retrieval_scores, "hit", normalized_k_values),
        mrr=_mean(score["mrr"] for score in retrieval_scores),
        precision_at_k=_aggregate_at_k(retrieval_scores, "precision", normalized_k_values),
        recall_at_k=_aggregate_at_k(retrieval_scores, "recall", normalized_k_values),
        ndcg_at_k=_aggregate_at_k(retrieval_scores, "ndcg", normalized_k_values),
        context_precision=_mean(score["precision"] for score in retrieval_scores),
        context_recall=_mean(score["recall"] for score in retrieval_scores),
        answer_exact_match=_mean(score["exact_match"] for score in answer_scores),
        answer_token_f1=_mean(score["token_f1"] for score in answer_scores),
    )


def _retrieval_scores(sample: EvaluationSample, k_values: Sequence[int]) -> dict[str, float] | None:
    reference_ids = sample.expected_chunk_ids or sample.expected_source_names
    retrieved_ids = sample.retrieved_chunk_ids or sample.retrieved_source_names
    if not reference_ids or not retrieved_ids:
        return None

    reference_set = set(reference_ids)
    relevance_flags = _binary_relevance_flags(retrieved_ids, reference_set)
    relevant_positions = [index for index, is_relevant in enumerate(relevance_flags, 1) if is_relevant]
    relevant_count = sum(relevance_flags)
    scores = {
        "hit": 1.0 if relevant_count else 0.0,
        "mrr": 1.0 / relevant_positions[0] if relevant_positions else 0.0,
        "precision": relevant_count / len(retrieved_ids),
        "recall": relevant_count / len(reference_set),
    }
    for k in k_values:
        top_k_flags = relevance_flags[:k]
        top_k_relevant_count = sum(top_k_flags)
        scores[f"hit@{k}"] = 1.0 if top_k_relevant_count else 0.0
        scores[f"precision@{k}"] = top_k_relevant_count / k
        scores[f"recall@{k}"] = top_k_relevant_count / len(reference_set)
        scores[f"ndcg@{k}"] = _ndcg_at_k(top_k_flags, len(reference_set), k)
    return scores


def _binary_relevance_flags(retrieved_ids: Sequence[str], reference_set: set[str]) -> list[int]:
    """Return binary relevance flags while ignoring duplicate relevant IDs."""
    seen_relevant: set[str] = set()
    flags: list[int] = []
    for item in retrieved_ids:
        if item in reference_set and item not in seen_relevant:
            flags.append(1)
            seen_relevant.add(item)
        else:
            flags.append(0)
    return flags


def _ndcg_at_k(relevance_flags: Sequence[int], reference_count: int, k: int) -> float:
    """Compute binary nDCG@k."""
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance_flags[:k]))
    ideal_relevance_count = min(reference_count, k)
    if ideal_relevance_count == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_relevance_count))
    return dcg / ideal_dcg


def _answer_scores(sample: EvaluationSample) -> dict[str, float] | None:
    if sample.expected_answer is None or sample.actual_answer is None:
        return None
    return {
        "exact_match": 1.0
        if _normalize_answer(sample.expected_answer) == _normalize_answer(sample.actual_answer)
        else 0.0,
        "token_f1": _token_f1(sample.expected_answer, sample.actual_answer),
    }


def _token_f1(expected: str, actual: str) -> float:
    expected_tokens = _tokenize(expected)
    actual_tokens = _tokenize(actual)
    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0

    overlap = Counter(expected_tokens) & Counter(actual_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0
    precision = shared / len(actual_tokens)
    recall = shared / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_answer(value: str) -> str:
    without_punctuation = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(without_punctuation.casefold().split())


def _tokenize(value: str) -> list[str]:
    return re.findall(r"\w+", _normalize_answer(value), flags=re.UNICODE)


def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _aggregate_at_k(
    scores: Sequence[Mapping[str, float]],
    metric_name: str,
    k_values: Sequence[int],
) -> dict[int, float | None]:
    """Aggregate one metric family for each k."""
    return {k: _mean(score[f"{metric_name}@{k}"] for score in scores) for k in k_values}


def _normalize_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    """Normalize k values while preserving order."""
    normalized: list[int] = []
    for value in k_values:
        if value <= 0 or value in normalized:
            continue
        normalized.append(value)
    if not normalized:
        return DEFAULT_K_VALUES
    return tuple(normalized)


def _string_keyed_metrics(values: Mapping[int, float | None]) -> dict[str, float | None]:
    """Convert integer metric keys to JSON-friendly string keys."""
    return {str(key): value for key, value in values.items()}


def _read_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if isinstance(value, Mapping):
        return value
    return {}


def _read_sequence(data: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = data.get(key)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def _read_first_string(data: Mapping[str, Any], keys: Sequence[str]) -> str:
    value = _read_optional_string(data, keys)
    if value is None:
        msg = f"Evaluation row must contain one of these question fields: {', '.join(keys)}"
        raise ValueError(msg)
    return value


def _read_optional_string(data: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_string_tuple(data: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    for key in keys:
        value = data.get(key)
        values = _coerce_string_sequence(value)
        if values:
            return values
    return ()


def _extract_field_tuple(items: Sequence[Any], keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            for key in keys:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
                    break
    return tuple(values)


def _coerce_string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()
