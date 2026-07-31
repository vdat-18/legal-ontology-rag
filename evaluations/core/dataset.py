"""Evaluation samples and their mapping onto Ragas single-turn records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_QUESTION_KEYS = ("question", "query", "user_input", "input")
_RESPONSE_KEYS = ("response", "actual_answer", "answer", "generated_answer")
_REFERENCE_KEYS = ("reference", "expected_answer", "ground_truth", "ground_truth_answer")
_RETRIEVED_CONTEXT_KEYS = ("retrieved_contexts", "contexts", "retrieved_chunks")
_REFERENCE_CONTEXT_KEYS = ("reference_contexts", "expected_contexts", "ground_truth_contexts")
_CONTEXT_TEXT_KEYS = ("text", "content", "context", "chunk")


@dataclass(frozen=True, slots=True)
class EvalSample:
    """One question with its generated answer, retrieved contexts, and optional gold reference."""

    user_input: str
    response: str
    retrieved_contexts: tuple[str, ...] = ()
    reference: str | None = None
    reference_contexts: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> EvalSample:
        """Read one flexible prediction row into a sample."""
        return cls(
            user_input=_first_string(row, _QUESTION_KEYS),
            response=_first_string(row, _RESPONSE_KEYS),
            retrieved_contexts=_context_texts(row, _RETRIEVED_CONTEXT_KEYS),
            reference=_optional_string(row, _REFERENCE_KEYS),
            reference_contexts=_context_texts(row, _REFERENCE_CONTEXT_KEYS),
        )

    def to_ragas_record(self) -> dict[str, Any]:
        """Render the sample as a Ragas single-turn record, omitting absent gold fields."""
        record: dict[str, Any] = {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),
        }
        if self.reference is not None:
            record["reference"] = self.reference
        if self.reference_contexts:
            record["reference_contexts"] = list(self.reference_contexts)
        return record


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
