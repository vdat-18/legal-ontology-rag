"""Tests for multi-turn ground-truth export."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from legal_rag.benchmark_synthesis.candidate_generation import ClaimEvidence, RequiredClaim
from legal_rag.benchmark_synthesis.groundtruth_export import export_multi_turn_ground_truth
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import (
    ConversationTurnDraft,
    MultiTurnCandidateDraft,
)
from legal_rag.evaluation.metrics import expand_evaluation_row
from legal_rag.evaluation.runner import load_evaluation_records


def _turn(
    turn_number: int,
    dialogue_act: str,
    *,
    answerability: str = "answerable",
    missing_information: str | None = None,
) -> ConversationTurnDraft:
    """Build one generated turn anchored on one chunk."""
    return ConversationTurnDraft(
        turn_number=turn_number,
        dialogue_act=dialogue_act,
        context_dependency="standalone" if turn_number == 1 else "ellipsis",
        answerability=answerability,
        anchor_scenario_id=f"scenario-{turn_number}",
        query=f"Câu hỏi lượt {turn_number}",
        standalone_query=f"Câu hỏi đầy đủ lượt {turn_number}",
        reference_answer=f"Đáp án lượt {turn_number}",
        reasoning="Theo căn cứ của lượt.",
        missing_information=missing_information,
        required_claims=(
            RequiredClaim(
                claim=f"Ý bắt buộc lượt {turn_number}",
                fact_id=f"fact-{turn_number}",
                evidence=(ClaimEvidence(chunk_id=f"chunk-{turn_number}"),),
            ),
        ),
        gold_chunk_ids=(f"chunk-{turn_number}",),
        source_fact_ids=(f"fact-{turn_number}",),
    )


def test_export_multi_turn_ground_truth_targets_only_final_turn() -> None:
    """Earlier turns become history and only the final turn is evaluated."""
    conversation = MultiTurnCandidateDraft(
        conversation_id="conv-1",
        persona="tai_xe_ca_nhan",
        conversation_length=2,
        turns=(
            _turn(1, "inquiry"),
            _turn(
                2,
                "continuity",
                answerability="partially_answerable",
                missing_information="mức xử lý bổ sung",
            ),
        ),
        gold_chunk_ids=("chunk-1", "chunk-2"),
        generator_model="provider/test-model",
    )

    (record,) = export_multi_turn_ground_truth(
        (conversation,),
        bundle_kind_by_scenario_id={"scenario-2": "multi_hop"},
    )

    assert record["interaction_type"] == "multi_turn"
    assert record["conversation_id"] == "conv-1"
    assert record["question_id"] == "conv-1:2"
    assert record["target_turn_number"] == 2
    assert record["history"] == [{"question": "Câu hỏi lượt 1", "answer": "Đáp án lượt 1"}]
    assert record["question"] == "Câu hỏi lượt 2"
    assert record["standalone_question"] == "Câu hỏi đầy đủ lượt 2"
    assert record["evaluation_mode"] == "retrieval_required"
    assert record["expected_answer"] == "Đáp án lượt 2"
    assert record["expected_chunk_ids"] == ["chunk-2"]
    assert record["dialogue_act"] == "continuity"
    assert record["context_dependency"] == "ellipsis"
    assert record["missing_information"] == "mức xử lý bổ sung"
    assert record["required_claims"][0]["fact_id"] == "fact-2"
    assert expand_evaluation_row(record) == (record,)


def test_multi_turn_export_uses_only_target_claim_dates_for_temporal_scope(tmp_path: Path) -> None:
    """Target claims define compared versions without leaking history or unused scenario facts."""
    history = _turn(1, "inquiry")
    target = replace(
        _turn(2, "continuity"),
        required_claims=(
            RequiredClaim("Quy định cũ", "fact-old", (ClaimEvidence("chunk-old"),)),
            RequiredClaim("Quy định mới", "fact-new", (ClaimEvidence("chunk-new"),)),
        ),
        gold_chunk_ids=("chunk-old", "chunk-new"),
        source_fact_ids=("fact-old", "fact-new", "fact-unused"),
    )
    conversation = MultiTurnCandidateDraft(
        conversation_id="conv-temporal",
        persona="tai_xe_ca_nhan",
        conversation_length=2,
        turns=(history, target),
        gold_chunk_ids=("chunk-1", "chunk-old", "chunk-new"),
        generator_model="provider/test-model",
    )
    facts = {
        fact.fact_id: fact
        for fact in (
            LegalFact("fact-1", "history", date(2022, 1, 1), None, "history", (EvidenceReference("chunk-1"),)),
            LegalFact("fact-old", "old", date(2025, 1, 1), None, "old", (EvidenceReference("chunk-old"),)),
            LegalFact("fact-new", "new", date(2026, 3, 1), None, "new", (EvidenceReference("chunk-new"),)),
            LegalFact(
                "fact-unused",
                "unused",
                date(2024, 1, 1),
                None,
                "unused",
                (EvidenceReference("chunk-unused"),),
            ),
        )
    }

    [record] = export_multi_turn_ground_truth(
        (conversation,),
        facts,
        {"scenario-2": "temporal"},
    )
    [multi_hop_record] = export_multi_turn_ground_truth(
        (conversation,),
        facts,
        {"scenario-2": "multi_hop"},
    )
    output = tmp_path / "groundtruth.jsonl"
    output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    [loaded] = load_evaluation_records(output)

    assert record["query_dates"] == ["2025-01-01", "2026-03-01"]
    assert record["bundle_kind"] == "temporal"
    assert loaded.query_dates == (date(2025, 1, 1), date(2026, 3, 1))
    assert loaded.temporal_comparison is True
    assert multi_hop_record["bundle_kind"] == "multi_hop"
    assert multi_hop_record["query_dates"] == []
