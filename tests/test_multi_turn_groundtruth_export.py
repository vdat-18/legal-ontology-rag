"""Tests for multi-turn ground-truth export."""

from __future__ import annotations

from legal_rag.benchmark_synthesis.candidate_generation import ClaimEvidence, RequiredClaim
from legal_rag.benchmark_synthesis.groundtruth_export import export_multi_turn_ground_truth
from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import (
    ConversationTurnDraft,
    MultiTurnCandidateDraft,
)


def _turn(turn_number: int, dialogue_act: str) -> ConversationTurnDraft:
    """Build one generated turn anchored on one chunk."""
    return ConversationTurnDraft(
        turn_number=turn_number,
        dialogue_act=dialogue_act,
        context_dependency="standalone" if turn_number == 1 else "ellipsis",
        answerability="answerable",
        anchor_scenario_id=f"scenario-{turn_number}",
        query=f"Câu hỏi lượt {turn_number}",
        reference_answer=f"Đáp án lượt {turn_number}",
        reasoning="Theo fact của lượt.",
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


def test_export_multi_turn_ground_truth_keeps_turn_level_evidence() -> None:
    """Exported conversations keep per-turn questions, evidence, and dialogue metadata."""
    conversation = MultiTurnCandidateDraft(
        conversation_id="conv-1",
        persona="tai_xe_ca_nhan",
        conversation_length=2,
        turns=(_turn(1, "inquiry"), _turn(2, "continuity")),
        gold_chunk_ids=("chunk-1", "chunk-2"),
        generator_model="provider/test-model",
    )

    (record,) = export_multi_turn_ground_truth((conversation,))

    assert record["interaction_type"] == "multi_turn"
    assert record["conversation_id"] == "conv-1"
    assert record["expected_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert record["question"] == "Câu hỏi lượt 1"
    assert record["expected_answer"] == "Đáp án lượt 1"
    assert [turn["turn_number"] for turn in record["turns"]] == [1, 2]
    second = record["turns"][1]
    assert second["question_id"] == "conv-1:2"
    assert second["question"] == "Câu hỏi lượt 2"
    assert second["expected_chunk_ids"] == ["chunk-2"]
    assert second["dialogue_act"] == "continuity"
    assert second["context_dependency"] == "ellipsis"
    assert second["required_claims"][0]["fact_id"] == "fact-2"
