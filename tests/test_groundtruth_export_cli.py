"""Tests for the evaluation ground-truth export command."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag import groundtruth_export_cli
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.groundtruth_export import GROUND_TRUTH_EXPORT_SCHEMA_VERSION
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.benchmark_synthesis.writer import (
    read_candidates,
    read_multi_turn_candidates,
    read_multi_turn_candidates_for_export,
    write_candidates,
    write_legal_facts,
    write_scenarios,
)
from legal_rag.evaluation.metrics import load_jsonl_samples
from legal_rag.io_utils import write_jsonl_atomic


def _candidate() -> CandidateDraft:
    """Build the smallest complete candidate accepted by the persisted schema."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query="Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
        reference_answer="Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.",
        reasoning="Áp dụng mức phạt tại Điều 7.",
        required_claims=(),
        gold_chunk_ids=("chunk-1",),
        query_form="specific",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        answerability="answerable",
        reasoning_hops=1,
        source_fact_ids=("fact-1",),
        generator_model="provider/model",
    )


def _legacy_multi_turn_candidate() -> dict[str, object]:
    """Build a schema-1.0 conversation without fields introduced later."""
    turns = [
        {
            "turn_number": turn_number,
            "dialogue_act": "inquiry" if turn_number == 1 else "continuity",
            "context_dependency": "standalone" if turn_number == 1 else "anaphora",
            "answerability": "answerable" if turn_number == 1 else "partially_answerable",
            "anchor_scenario_id": f"scenario-{turn_number}",
            "query": f"Câu hỏi lượt {turn_number}",
            "reference_answer": f"Đáp án lượt {turn_number}",
            "reasoning": f"Lập luận lượt {turn_number}",
            "required_claims": [
                {
                    "claim": f"Ý bắt buộc lượt {turn_number}",
                    "fact_id": f"fact-{turn_number}",
                    "evidence": [{"chunk_id": f"chunk-{turn_number}"}],
                }
            ],
            "gold_chunk_ids": [f"chunk-{turn_number}"],
            "source_fact_ids": [f"fact-{turn_number}"],
        }
        for turn_number in (1, 2)
    ]
    return {
        "schema_version": "1.0.0",
        "interaction_type": "multi_turn",
        "conversation_id": "conversation-legacy",
        "persona": "tai_xe_ca_nhan",
        "conversation_length": 2,
        "turns": turns,
        "gold_chunk_ids": ["chunk-1", "chunk-2"],
        "generator_model": "provider/legacy-model",
        "prompt_version": "1.0.0",
    }


def test_groundtruth_export_cli_writes_evaluation_ready_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The CLI exports every candidate atomically in input order."""
    candidates_path = tmp_path / "single_turn.jsonl"
    output_path = tmp_path / "ground_truth.jsonl"
    write_candidates(candidates_path, (_candidate(),))
    assert len(read_candidates(candidates_path)) == 1
    monkeypatch.setattr(
        sys,
        "argv",
        ["legal-export-groundtruth", str(candidates_path), str(output_path)],
    )

    groundtruth_export_cli.main()

    samples = load_jsonl_samples(output_path)
    assert len(samples) == 1
    assert samples[0].question == _candidate().query
    assert f'"schema_version":"{GROUND_TRUTH_EXPORT_SCHEMA_VERSION}"' in output_path.read_text(encoding="utf-8")


def test_groundtruth_export_cli_uses_legal_facts_for_temporal_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI enriches temporal candidates from the selected legal-facts release."""
    candidates_path = tmp_path / "single_turn.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    output_path = tmp_path / "ground_truth.jsonl"
    write_candidates(candidates_path, (replace(_candidate(), bundle_kind="temporal"),))
    write_legal_facts(
        facts_path,
        (
            LegalFact(
                "fact-1",
                "rule",
                date(2025, 1, 1),
                None,
                "document",
                (EvidenceReference("chunk-1"),),
            ),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-export-groundtruth",
            str(candidates_path),
            str(output_path),
            "--legal-facts",
            str(facts_path),
        ],
    )

    groundtruth_export_cli.main()

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["query_dates"] == ["2025-01-01"]


def test_multi_turn_export_cli_accepts_legacy_candidates_without_inventing_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Schema 1.0 keeps gold history while unavailable later fields stay null."""
    candidates_path = tmp_path / "legacy_multi_turn.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    output_path = tmp_path / "ground_truth.jsonl"
    write_jsonl_atomic(candidates_path, (_legacy_multi_turn_candidate(),))
    write_scenarios(
        scenarios_path,
        (
            Scenario(
                scenario_id="scenario-2",
                query_form="specific",
                answerability="partially_answerable",
                persona="tai_xe_ca_nhan",
                query_style="standard",
                query_length="short",
                reasoning_hops=1,
                fact_ids=("fact-2",),
                bridge=None,
                gold_chunk_ids=("chunk-2",),
                bundle_kind="multi_hop",
            ),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-export-groundtruth",
            str(candidates_path),
            str(output_path),
            "--multi-turn",
            "--scenarios",
            str(scenarios_path),
        ],
    )

    groundtruth_export_cli.main()

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["conversation_id"] == "conversation-legacy"
    assert record["question_id"] == "conversation-legacy:2"
    assert record["target_turn_number"] == 2
    assert record["history"] == [{"question": "Câu hỏi lượt 1", "answer": "Đáp án lượt 1"}]
    assert record["question"] == "Câu hỏi lượt 2"
    assert record["standalone_question"] is None
    assert record["missing_information"] is None
    assert record["expected_chunk_ids"] == ["chunk-2"]
    (sample,) = load_jsonl_samples(output_path)
    assert sample.history == (("Câu hỏi lượt 1", "Đáp án lượt 1"),)


def test_legacy_multi_turn_compatibility_is_limited_to_export(tmp_path: Path) -> None:
    """The generation reader remains strict while the export reader accepts 1.0."""
    candidates_path = tmp_path / "legacy_multi_turn.jsonl"
    write_jsonl_atomic(candidates_path, (_legacy_multi_turn_candidate(),))

    with pytest.raises(ValueError, match=r"Unsupported multi-turn candidate schema: 1\.0\.0"):
        read_multi_turn_candidates(candidates_path)

    (conversation,) = read_multi_turn_candidates_for_export(candidates_path)
    assert conversation.turns[-1].standalone_query == ""
    assert conversation.turns[-1].missing_information is None


def test_export_reader_keeps_schema_4_fields_strict(tmp_path: Path) -> None:
    """Legacy defaults must not mask missing fields in current-schema candidates."""
    payload = _legacy_multi_turn_candidate()
    payload["schema_version"] = "4.0.0"
    candidates_path = tmp_path / "invalid_current_multi_turn.jsonl"
    write_jsonl_atomic(candidates_path, (payload,))

    with pytest.raises(KeyError, match="standalone_query"):
        read_multi_turn_candidates_for_export(candidates_path)
