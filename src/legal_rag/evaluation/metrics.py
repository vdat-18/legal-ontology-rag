"""Offline metrics for Legal RAG evaluation datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_SLICE_FIELDS = ("bundle_kind", "answerability", "reasoning_hops", "query_style")


@dataclass(frozen=True, slots=True)
class EvaluationRetrievedUnit:
    """One pipeline-native retrieval unit and its comparable source evidence."""

    unit_id: str
    evidence_chunk_ids: tuple[str, ...] = ()
    evidence_source_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpectedClaimEvidence:
    """Gold evidence required to support one expected answer claim."""

    claim: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """One answer/retrieval evaluation sample."""

    question: str
    bundle_kind: str | None = None
    answerability: str | None = None
    reasoning_hops: int | None = None
    query_style: str | None = None
    expected_answer: str | None = None
    actual_answer: str | None = None
    expected_chunk_ids: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    expected_source_names: tuple[str, ...] = ()
    retrieved_source_names: tuple[str, ...] = ()
    retrieved_units: tuple[EvaluationRetrievedUnit, ...] = ()
    expected_claims: tuple[ExpectedClaimEvidence, ...] = ()
    history: tuple[tuple[str, str], ...] = ()

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
        history = _read_sequence(data, "history")

        return cls(
            question=_read_first_string(data, ("question", "query", "user_input", "input")),
            bundle_kind=_read_optional_string(data, ("bundle_kind",)),
            answerability=_read_optional_string(data, ("answerability",)),
            reasoning_hops=_read_optional_int(data, "reasoning_hops"),
            query_style=_read_optional_string(data, ("query_style",)),
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
            retrieved_units=_read_retrieved_units(data),
            expected_claims=_read_expected_claims(data),
            history=tuple(
                (item["question"], item["answer"])
                for item in history
                if isinstance(item, Mapping)
                and isinstance(item.get("question"), str)
                and isinstance(item.get("answer"), str)
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate evaluation metrics."""

    sample_count: int
    retrieval_sample_count: int
    claim_sample_count: int
    k_values: tuple[int, ...]
    hit_rate: float | None
    hit_rate_at_k: dict[int, float | None]
    precision_at_k: dict[int, float | None]
    recall_at_k: dict[int, float | None]
    context_precision: float | None
    context_recall: float | None
    empty_retrieval_rate: float | None
    claim_evidence_any_at_k: dict[int, float | None]
    claim_evidence_complete_at_k: dict[int, float | None]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "sample_count": self.sample_count,
            "retrieval_sample_count": self.retrieval_sample_count,
            "claim_sample_count": self.claim_sample_count,
            "k_values": list(self.k_values),
            "hit_rate": self.hit_rate,
            "hit_rate_at_k": _string_keyed_metrics(self.hit_rate_at_k),
            "precision_at_k": _string_keyed_metrics(self.precision_at_k),
            "recall_at_k": _string_keyed_metrics(self.recall_at_k),
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "empty_retrieval_rate": self.empty_retrieval_rate,
            "claim_evidence_any_at_k": _string_keyed_metrics(self.claim_evidence_any_at_k),
            "claim_evidence_complete_at_k": _string_keyed_metrics(self.claim_evidence_complete_at_k),
        }


def load_jsonl_samples(path: Path) -> list[EvaluationSample]:
    """Load flat samples and expand nested multi-turn conversations by turn."""
    samples: list[EvaluationSample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                msg = f"Expected JSON object at {path}:{line_number}"
                raise TypeError(msg)
            samples.extend(EvaluationSample.from_mapping(item) for item in expand_evaluation_row(row))
    return samples


def expand_evaluation_row(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Expand one nested multi-turn ground-truth row into turn-level samples."""
    turns = data.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return (data,)
    expanded: list[Mapping[str, Any]] = []
    history: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            msg = "Multi-turn ground truth turns must be JSON objects"
            raise TypeError(msg)
        row = dict(data)
        row.pop("turns", None)
        row.update(turn)
        row["history"] = list(history)
        expanded.append(row)
        question = turn.get("question")
        answer = turn.get("expected_answer")
        if isinstance(question, str) and isinstance(answer, str):
            history.append({"question": question, "answer": answer})
    return tuple(expanded)


def evaluate_samples(
    samples: Sequence[EvaluationSample],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> EvaluationMetrics:
    """Compute aggregate native-unit retrieval and evidence-coverage metrics."""
    normalized_k_values = _normalize_k_values(k_values)
    retrieval_scores = [_retrieval_scores(sample, normalized_k_values) for sample in samples]
    retrieval_scores = [score for score in retrieval_scores if score is not None]
    claim_scores = [score for score in retrieval_scores if score["has_claims"]]

    return EvaluationMetrics(
        sample_count=len(samples),
        retrieval_sample_count=len(retrieval_scores),
        claim_sample_count=len(claim_scores),
        k_values=normalized_k_values,
        hit_rate=_mean(score["hit"] for score in retrieval_scores),
        hit_rate_at_k=_aggregate_at_k(retrieval_scores, "hit", normalized_k_values),
        precision_at_k=_aggregate_at_k(retrieval_scores, "precision", normalized_k_values),
        recall_at_k=_aggregate_at_k(retrieval_scores, "recall", normalized_k_values),
        context_precision=_mean(score["precision"] for score in retrieval_scores),
        context_recall=_mean(score["recall"] for score in retrieval_scores),
        empty_retrieval_rate=_mean(score["empty"] for score in retrieval_scores),
        claim_evidence_any_at_k=_aggregate_at_k(claim_scores, "claim_any", normalized_k_values),
        claim_evidence_complete_at_k=_aggregate_at_k(claim_scores, "claim_complete", normalized_k_values),
    )


def evaluate_slices(
    rows: Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    fields: Sequence[str] = DEFAULT_SLICE_FIELDS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate configured metadata slices without discarding the shared metrics."""
    slices: dict[str, dict[str, dict[str, Any]]] = {}
    for field in fields:
        groups: dict[str, list[EvaluationSample]] = {}
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            groups.setdefault(str(value), []).append(EvaluationSample.from_mapping(row))
        if groups:
            slices[field] = {
                value: evaluate_samples(samples, k_values=k_values).to_dict()
                for value, samples in sorted(groups.items())
            }
    return slices


def _retrieval_scores(sample: EvaluationSample, k_values: Sequence[int]) -> dict[str, float | bool] | None:
    label_pair = _retrieval_label_pair(sample)
    if label_pair is None:
        return None

    reference_kind, reference_ids = label_pair
    reference_set = set(reference_ids)
    units = _effective_units(sample)
    unit_evidence = [
        set(unit.evidence_chunk_ids if reference_kind == "chunk" else unit.evidence_source_names) for unit in units
    ]
    relevance_flags = [1 if evidence & reference_set else 0 for evidence in unit_evidence]
    covered = set().union(*unit_evidence) & reference_set if unit_evidence else set()
    relevant_count = sum(relevance_flags)
    scores: dict[str, float | bool] = {
        "hit": 1.0 if relevant_count else 0.0,
        "precision": relevant_count / len(units) if units else 0.0,
        "recall": len(covered) / len(reference_set),
        "empty": 0.0 if units else 1.0,
        "has_claims": bool(sample.expected_claims),
    }
    for k in k_values:
        top_k_evidence = set().union(*unit_evidence[:k]) if unit_evidence[:k] else set()
        top_k_relevant_count = sum(relevance_flags[:k])
        scores[f"hit@{k}"] = 1.0 if top_k_relevant_count else 0.0
        scores[f"precision@{k}"] = top_k_relevant_count / k
        scores[f"recall@{k}"] = len(top_k_evidence & reference_set) / len(reference_set)
        claim_any, claim_complete = _claim_coverage(sample.expected_claims, top_k_evidence)
        scores[f"claim_any@{k}"] = claim_any
        scores[f"claim_complete@{k}"] = claim_complete
    return scores


def _effective_units(sample: EvaluationSample) -> tuple[EvaluationRetrievedUnit, ...]:
    if sample.retrieved_units:
        return sample.retrieved_units
    return tuple(
        EvaluationRetrievedUnit(
            unit_id=sample.retrieved_chunk_ids[index]
            if index < len(sample.retrieved_chunk_ids)
            else f"source:{sample.retrieved_source_names[index]}",
            evidence_chunk_ids=(sample.retrieved_chunk_ids[index],) if index < len(sample.retrieved_chunk_ids) else (),
            evidence_source_names=(sample.retrieved_source_names[index],)
            if index < len(sample.retrieved_source_names)
            else (),
        )
        for index in range(max(len(sample.retrieved_chunk_ids), len(sample.retrieved_source_names)))
    )


def _claim_coverage(claims: Sequence[ExpectedClaimEvidence], covered_chunk_ids: set[str]) -> tuple[float, float]:
    if not claims:
        return 0.0, 0.0
    any_count = sum(bool(set(claim.chunk_ids) & covered_chunk_ids) for claim in claims)
    complete_count = sum(bool(claim.chunk_ids) and set(claim.chunk_ids) <= covered_chunk_ids for claim in claims)
    return any_count / len(claims), complete_count / len(claims)


def _retrieval_label_pair(sample: EvaluationSample) -> tuple[str, tuple[str, ...]] | None:
    if sample.expected_chunk_ids:
        return "chunk", sample.expected_chunk_ids
    if sample.expected_source_names:
        return "source", sample.expected_source_names
    return None


def _read_retrieved_units(data: Mapping[str, Any]) -> tuple[EvaluationRetrievedUnit, ...]:
    units: list[EvaluationRetrievedUnit] = []
    for item in _read_sequence(data, "retrieved_units"):
        if not isinstance(item, Mapping):
            continue
        unit_id = _read_optional_string(item, ("unit_id", "id"))
        if unit_id is None:
            continue
        evidence = item.get("evidence")
        evidence_items = evidence if isinstance(evidence, Sequence) and not isinstance(evidence, str) else ()
        units.append(
            EvaluationRetrievedUnit(
                unit_id=unit_id,
                evidence_chunk_ids=_extract_field_tuple(evidence_items, ("chunk_id", "id")),
                evidence_source_names=_extract_field_tuple(evidence_items, ("source_name", "source", "name")),
            )
        )
    return tuple(units)


def _read_expected_claims(data: Mapping[str, Any]) -> tuple[ExpectedClaimEvidence, ...]:
    claims: list[ExpectedClaimEvidence] = []
    for item in _read_sequence(data, "required_claims"):
        if not isinstance(item, Mapping):
            continue
        claim = _read_optional_string(item, ("claim",))
        evidence = item.get("evidence")
        evidence_items = evidence if isinstance(evidence, Sequence) and not isinstance(evidence, str) else ()
        chunk_ids = _extract_field_tuple(evidence_items, ("chunk_id", "id"))
        if claim is not None and chunk_ids:
            claims.append(ExpectedClaimEvidence(claim=claim, chunk_ids=chunk_ids))
    return tuple(claims)


def _mean(values: Iterable[float | bool]) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _aggregate_at_k(
    scores: Sequence[Mapping[str, float | bool]],
    metric_name: str,
    k_values: Sequence[int],
) -> dict[int, float | None]:
    return {k: _mean(score[f"{metric_name}@{k}"] for score in scores) for k in k_values}


def _normalize_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in k_values:
        if value > 0 and value not in normalized:
            normalized.append(value)
    return tuple(normalized) or DEFAULT_K_VALUES


def _string_keyed_metrics(values: Mapping[int, float | None]) -> dict[str, float | None]:
    return {str(key): value for key, value in values.items()}


def _read_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


def _read_sequence(data: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = data.get(key)
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


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


def _read_optional_int(data: Mapping[str, Any], key: str) -> int | None:
    """Read one optional integer field without accepting booleans as integers."""
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_string_tuple(data: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    for key in keys:
        values = _coerce_string_sequence(data.get(key))
        if values:
            return values
    return ()


def _extract_field_tuple(items: Sequence[Any], keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            value = _read_optional_string(item, keys)
            if value is not None:
                values.append(value)
    return tuple(values)


def _coerce_string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()
