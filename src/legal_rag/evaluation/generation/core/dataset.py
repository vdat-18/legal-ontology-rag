"""Evaluation samples and their flexible JSONL mapping."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

_QUESTION_KEYS = ("question", "query", "user_input", "input")
_RESPONSE_KEYS = ("response", "actual_answer", "answer", "generated_answer")
_REFERENCE_KEYS = (
    "reference",
    "expected_answer",
    "reference_answer",
    "ground_truth",
    "ground_truth_answer",
)
_RETRIEVED_CONTEXT_KEYS = ("retrieved_contexts", "contexts", "retrieved_chunks")
_REFERENCE_CONTEXT_KEYS = ("reference_contexts", "expected_contexts", "ground_truth_contexts")
_CONTEXT_TEXT_KEYS = ("text", "content", "context", "chunk")
_SAMPLE_ID_KEYS = ("evaluation_record_id", "question_id", "sample_id")
_HISTORY_KEYS = ("history", "conversation_history")
_HISTORY_QUESTION_KEYS = ("question", "query", "user_input")
_HISTORY_ANSWER_KEYS = (
    "answer",
    "response",
    "actual_answer",
    "expected_answer",
    "reference_answer",
)


@dataclass(frozen=True, slots=True)
class EvalSample:
    """One question with its generated answer, retrieved contexts, and optional gold reference."""

    user_input: str
    response: str
    sample_id: str = ""
    retrieved_contexts: tuple[str, ...] = ()
    reference: str | None = None
    reference_contexts: tuple[str, ...] = ()
    history: tuple[tuple[str, str], ...] = ()
    answerability: str | None = None
    required_claims: tuple[str, ...] = ()
    query_date: date | None = None
    query_dates: tuple[date, ...] = ()
    reference_date: date | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> EvalSample:
        """Read one flexible prediction row into a sample."""
        query_date = _optional_date(row, "query_date")
        query_dates = _optional_dates(row, "query_dates")
        if query_date is not None and query_dates and query_date not in query_dates:
            msg = "query_date must be included in query_dates"
            raise ValueError(msg)
        return cls(
            user_input=_first_string(row, _QUESTION_KEYS),
            response=_first_string(row, _RESPONSE_KEYS),
            sample_id=_optional_string(row, _SAMPLE_ID_KEYS) or "",
            retrieved_contexts=_context_texts(row, _RETRIEVED_CONTEXT_KEYS),
            reference=_optional_string(row, _REFERENCE_KEYS),
            reference_contexts=_context_texts(row, _REFERENCE_CONTEXT_KEYS),
            history=_history_turns(row),
            answerability=_optional_string(row, ("answerability",)),
            required_claims=_required_claim_texts(row),
            query_date=query_date,
            query_dates=query_dates,
            reference_date=_optional_date(row, "reference_date"),
        )


def load_samples(path: Path) -> list[EvalSample]:
    """Load evaluation samples from a JSONL predictions file."""
    samples: list[EvalSample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                msg = f"Expected a JSON object at {path}:{line_number}"
                raise TypeError(msg)
            samples.append(EvalSample.from_mapping(row))
    return samples


def _optional_string(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    """Return the first non-empty string among *keys*, or None."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_string(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first non-empty string among *keys*, or raise when none is present."""
    value = _optional_string(row, keys)
    if value is None:
        msg = f"Row must contain one of these fields: {', '.join(keys)}"
        raise ValueError(msg)
    return value


def _optional_date(row: Mapping[str, Any], key: str) -> date | None:
    """Read one optional canonical ISO date without consulting the wall clock."""
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        msg = f"{key} must be a date or canonical YYYY-MM-DD string"
        raise TypeError(msg)
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        msg = f"{key} must be a date or canonical YYYY-MM-DD string"
        raise TypeError(msg)
    text = value.strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        msg = f"{key} must use canonical YYYY-MM-DD format"
        raise ValueError(msg) from error
    if text != parsed.isoformat():
        msg = f"{key} must use canonical YYYY-MM-DD format"
        raise ValueError(msg)
    return parsed


def _optional_dates(row: Mapping[str, Any], key: str) -> tuple[date, ...]:
    """Read a unique sequence of canonical ISO dates without consulting the wall clock."""
    value = row.get(key)
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"{key} must be an array of dates or canonical YYYY-MM-DD strings"
        raise TypeError(msg)
    parsed = tuple(_optional_date({key: item}, key) for item in value)
    if any(item is None for item in parsed):
        msg = f"{key} must not contain null or empty values"
        raise ValueError(msg)
    dates = tuple(item for item in parsed if item is not None)
    if len(set(dates)) != len(dates):
        msg = f"{key} must not contain duplicates"
        raise ValueError(msg)
    return dates


def _context_texts(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    """Return context texts from the first present key, accepting strings or text-bearing objects."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str):
            texts = tuple(text for item in value if (text := _context_text(item)))
            if texts:
                return texts
    return ()


def _context_text(item: Any) -> str | None:
    """Extract one context text from a plain string or a text-bearing mapping."""
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, Mapping):
        for key in _CONTEXT_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _required_claim_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the expected claim text carried by synthesized ground truth."""
    value = row.get("required_claims")
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    claims: list[str] = []
    for item in value:
        claim = item.get("claim") if isinstance(item, Mapping) else item
        if isinstance(claim, str) and claim.strip():
            claims.append(claim.strip())
    return tuple(claims)


def _history_turns(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Read earlier question-answer pairs while accepting legacy tuple-shaped turns."""
    for key in _HISTORY_KEYS:
        value = row.get(key)
        if not isinstance(value, Sequence) or isinstance(value, str):
            continue
        turns: list[tuple[str, str]] = []
        for item in value:
            turn = _history_turn(item)
            if turn is not None:
                turns.append(turn)
        return tuple(turns)
    return ()


def _history_turn(item: Any) -> tuple[str, str] | None:
    """Normalize one mapping- or pair-shaped history turn."""
    if isinstance(item, Mapping):
        question = _optional_string(item, _HISTORY_QUESTION_KEYS)
        answer = _optional_string(item, _HISTORY_ANSWER_KEYS)
    elif isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 2:
        question = item[0].strip() if isinstance(item[0], str) and item[0].strip() else None
        answer = item[1].strip() if isinstance(item[1], str) and item[1].strip() else None
    else:
        return None
    return (question, answer) if question is not None and answer is not None else None
