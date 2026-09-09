"""Tests for reading evaluation samples and mapping them onto Ragas records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.evaluation.generation.core.dataset import EvalSample, load_samples


def test_from_mapping_reads_canonical_fields_with_string_contexts() -> None:
    """Canonical field names and plain-string contexts are read directly."""
    sample = EvalSample.from_mapping(
        {
            "question": "vượt đèn đỏ phạt bao nhiêu?",
            "response": "Phạt 4-6 triệu.",
            "retrieved_contexts": ["Điều 6 ...", "  "],
            "reference": "Phạt tiền từ 4 đến 6 triệu đồng.",
        }
    )

    assert sample.user_input == "vượt đèn đỏ phạt bao nhiêu?"
    assert sample.response == "Phạt 4-6 triệu."
    assert sample.retrieved_contexts == ("Điều 6 ...",)
    assert sample.reference == "Phạt tiền từ 4 đến 6 triệu đồng."
    assert sample.reference_contexts == ()


def test_from_mapping_reads_alternate_field_names_and_object_contexts() -> None:
    """Alternate keys and text-bearing context objects are supported."""
    sample = EvalSample.from_mapping(
        {
            "query": "câu hỏi",
            "actual_answer": "trả lời",
            "contexts": [{"text": "ngữ cảnh 1"}, {"content": "ngữ cảnh 2"}],
            "expected_answer": "đáp án",
            "expected_contexts": [{"chunk_id": "c1", "text": "gold 1"}],
        }
    )

    assert sample.user_input == "câu hỏi"
    assert sample.response == "trả lời"
    assert sample.retrieved_contexts == ("ngữ cảnh 1", "ngữ cảnh 2")
    assert sample.reference == "đáp án"
    assert sample.reference_contexts == ("gold 1",)


def test_from_mapping_raises_when_the_question_field_is_missing() -> None:
    """A row with no recognizable question field fails loudly."""
    with pytest.raises(ValueError, match="question"):
        EvalSample.from_mapping({"response": "a"})


def test_load_samples_reads_one_sample_per_nonblank_line(tmp_path: Path) -> None:
    """JSONL rows load in order, skipping blank lines."""
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"question": "q1", "response": "a1"})
        + "\n\n"
        + json.dumps({"question": "q2", "response": "a2"})
        + "\n",
        encoding="utf-8",
    )

    samples = load_samples(path)

    assert [sample.user_input for sample in samples] == ["q1", "q2"]
