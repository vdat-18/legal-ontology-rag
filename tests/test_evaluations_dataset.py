"""Tests for reading evaluation samples and mapping them onto Ragas records."""

from __future__ import annotations

import json
from datetime import date
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


def test_from_mapping_accepts_reference_answer_alias() -> None:
    """Legacy rows may carry the gold answer under reference_answer."""
    sample = EvalSample.from_mapping(
        {
            "question": "câu hỏi",
            "response": "trả lời",
            "reference_answer": "đáp án tham chiếu",
        }
    )

    assert sample.reference == "đáp án tham chiếu"


def test_from_mapping_reads_stable_identity_answerability_and_claims() -> None:
    """Generation judging retains benchmark fields needed for traceable analysis."""
    sample = EvalSample.from_mapping(
        {
            "question_id": "question-1",
            "question": "câu hỏi",
            "actual_answer": "trả lời",
            "answerability": "partially_answerable",
            "required_claims": [
                {"claim": "  claim one  ", "evidence": [{"chunk_id": "chunk-1"}]},
                {"claim": "", "evidence": []},
            ],
        }
    )

    assert sample.sample_id == "question-1"
    assert sample.answerability == "partially_answerable"
    assert sample.required_claims == ("claim one",)


def test_from_mapping_preserves_temporal_evaluation_dates() -> None:
    """Per-row query dates and frozen campaign dates remain distinct for judging."""
    sample = EvalSample.from_mapping(
        {
            "question": "câu hỏi",
            "response": "trả lời",
            "query_date": "2025-01-01",
            "query_dates": ["2025-01-01", "2026-03-01"],
            "reference_date": date(2026, 8, 31),
        }
    )

    assert sample.query_date == date(2025, 1, 1)
    assert sample.query_dates == (date(2025, 1, 1), date(2026, 3, 1))
    assert sample.reference_date == date(2026, 8, 31)


@pytest.mark.parametrize("field", ["query_date", "reference_date"])
def test_from_mapping_rejects_noncanonical_temporal_date(field: str) -> None:
    """Invalid temporal anchors fail instead of silently falling back to the wall clock."""
    with pytest.raises(ValueError, match=field):
        EvalSample.from_mapping({"question": "câu hỏi", "response": "trả lời", field: "20250101"})


def test_from_mapping_rejects_conflicting_singular_and_plural_dates() -> None:
    """A row cannot declare two incompatible temporal contracts."""
    with pytest.raises(ValueError, match="included in query_dates"):
        EvalSample.from_mapping(
            {
                "question": "câu hỏi",
                "response": "trả lời",
                "query_date": "2025-01-01",
                "query_dates": ["2026-03-01"],
            }
        )


def test_from_mapping_reads_conversation_history_and_legacy_claim_strings() -> None:
    """Multi-turn context reaches the judge while older string claim rows remain readable."""
    sample = EvalSample.from_mapping(
        {
            "question": "Còn xe máy thì sao?",
            "response": "Xe máy bị phạt...",
            "history": [
                {"question": "Ô tô vượt đèn đỏ bị phạt bao nhiêu?", "answer": "Phạt 18-20 triệu đồng."},
                ["Có bị trừ điểm không?", "Có, bị trừ 4 điểm."],
                {"question": "", "answer": "malformed"},
            ],
            "required_claims": ["  Mức phạt tiền  ", {"claim": "Số điểm bị trừ"}],
        }
    )

    assert sample.history == (
        ("Ô tô vượt đèn đỏ bị phạt bao nhiêu?", "Phạt 18-20 triệu đồng."),
        ("Có bị trừ điểm không?", "Có, bị trừ 4 điểm."),
    )
    assert sample.required_claims == ("Mức phạt tiền", "Số điểm bị trừ")


def test_from_mapping_accepts_conversation_history_alias() -> None:
    """The legacy conversation_history key remains supported."""
    sample = EvalSample.from_mapping(
        {
            "question": "follow-up",
            "response": "answer",
            "conversation_history": [
                {"query": "first", "expected_answer": "first answer"},
                {"query": "second", "reference_answer": "second answer"},
            ],
        }
    )

    assert sample.history == (("first", "first answer"), ("second", "second answer"))


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
